---
title: "De Chatbot a Agente: Construindo um Assistente PIX Conversacional com LangGraph e FastAPI"
date: 2026-07-16
draft: false
tags: ["langgraph", "fastapi", "llm", "ai-engineering", "python", "openrouter", "ollama", "observability"]
img: langgraph.png
categories: ["AI Engineering"]
---

## A Lacuna Entre Chatbots e Agentes

A maioria das demos de LLM para exatamente onde as coisas começam a ficar interessantes: o modelo gera um texto bonito e é isso. Mas o valor real de um LLM dentro de um produto não está no que ele *diz* — está no que ele *faz* depois de falar.

Nos últimos meses, venho construindo um projeto paralelo que cruza essa linha de propósito: um **assistente conversacional de PIX** que interpreta comandos em linguagem natural e executa operações bancárias reais. O usuário digita _"lista minhas chaves pix"_ ou _"paga esse QR Code"_, e o sistema classifica a intenção, roteia por um grafo de estados, chama a API bancária e responde com uma mensagem humanizada — com autenticação, cache, persistência, guardrails e observabilidade completa no caminho.

Este post percorre a arquitetura e os trade-offs por trás de cada decisão. Todos os trechos de código são reais, tirados diretamente do repositório.

## Contexto: Da Sala de Aula para o Projeto Paralelo

O timing não foi acidental. Estou cursando uma pós-graduação em **Engenharia de IA Aplicada**, e o módulo de integração com APIs de LLM (ministrado pelo [Erick Wendel](https://github.com/ErickWendel), que incentiva ativamente os alunos a construir em público) apresentou exatamente as ferramentas que eu queria testar na prática: **LangGraph** para orquestração de agentes e **OpenRouter** para acesso multi-modelo.

Ao mesmo tempo, meu trabalho do dia a dia é construir produtos de Banking-as-a-Service. Combinar os dois mundos foi o movimento natural: pegar os padrões de agentes da sala de aula e testá-los contra o tipo de complexidade de integração que vejo todos os dias em banking — autenticação JWT, idempotência, estado transacional, trilhas de auditoria.

O resultado são dois repositórios:

- **[banking-llm](https://github.com/mabittar/banking-llm)** — o backend: um serviço FastAPI hospedando uma máquina de estados LangGraph
- **[chat-llm](https://github.com/mabittar/chat-llm)** — o frontend: uma SPA de chat em Vue 3 que conversa com ele

## O Que o Sistema Faz

Um único endpoint, `POST /chat`, recebe uma mensagem em linguagem natural e retorna uma resposta humanizada. Por trás dele, seis intenções são suportadas:

| Intenção         | Descrição                                                |
|------------------|----------------------------------------------------------|
| `list_keys`      | Lista chaves PIX ativas da conta                         |
| `read_key`       | Consulta detalhes de uma chave PIX específica            |
| `pix_withdraw`   | Executa transferência PIX para uma chave                 |
| `brcode_preview` | Decodifica e valida um QR Code PIX                       |
| `pix_payment`    | Pagamento completo via QR Code (preview + transferência) |
| `guardrail`      | Validação de segurança contra prompt injection           |

## Visão Geral da Arquitetura

O serviço segue um design em camadas, com a máquina LangGraph entre a camada HTTP e os serviços de domínio:

```
┌─────────────────────────────────────────────────────────┐
│                   HTTP Layer (FastAPI)                  │
│  POST /chat  →  ChatRequest → GraphProcessor.ainvoke()  │
├─────────────────────────────────────────────────────────┤
│                   Graph Layer (LangGraph)               │
│  StateGraph[GraphState]                                 │
│    ├── guardrail        (Prompt injection guard)        │
│    ├── identifyIntent   (LLM-as-router)                 │
│    ├── listKeys         (Banking API — list keys)       │
│    ├── readKey          (Banking API — key details)     │
│    ├── pixWithdraw      (Banking API — PIX transfer)    │
│    ├── brcodePreview    (Banking API — QR decode)       │
│    ├── pixPayment       (Orchestrator: preview + pay)   │
│    ├── fallback         (No-op handler)                 │
│    └── chatResponse     (LLM-as-generator)              │
├─────────────────────────────────────────────────────────┤
│                   Services Layer                        │
│    Guardrail · Intent · PixKey · PixWithdraw ·          │
│    BRCodePreview · PixPayment · Response                │
├─────────────────────────────────────────────────────────┤
│               Infrastructure Layer                      │
│    LLMService (Ollama/OpenRouter) · BankingClient       │
│    BankingAuth (JWT ES512) · RedisCacheService          │
│    AsyncPostgresSaver (LangGraph persistence)           │
└─────────────────────────────────────────────────────────┘
```

Agora vamos às decisões que realmente importaram.

## Decisão 1: Máquina de Estados, Não um Agente Livre

A primeira bifurcação arquitetural: deve ser um agente estilo ReAct que escolhe ferramentas autonomamente, ou um grafo de estados explícito?

| Abordagem                        | Prós                                                                                              | Contras                                                          | Decisão     |
|----------------------------------|---------------------------------------------------------------------------------------------------|------------------------------------------------------------------|-------------|
| **Máquina de estados LangGraph** | Roteamento determinístico, caminho auditável, observabilidade por nó, fácil de reproduzir/debugar | Mais design inicial, comportamento menos "emergente"             | ✅ Escolhida |
| Agente ReAct (tool calling)      | Flexível, menos código, lida com caminhos não planejados                                          | Não determinístico, difícil de auditar, difícil garantir o fluxo | ❌           |
| Encadeamento puro de prompts     | Simples                                                                                           | Sem lógica condicional, frágil                                   | ❌           |

**O insight-chave:** quando a "ferramenta" do outro lado é uma API real de movimentação financeira, *não determinismo não é uma feature — é um passivo*. Preciso garantir, por construção, que toda requisição passe pelo guardrail antes de qualquer coisa tocar a API bancária, e que um pagamento só aconteça depois de um preview. Um grafo de estados me dá essas garantias como arestas, não como esperanças embutidas num system prompt.

O fluxo é explícito:

```
START ──▶ guardrail ──▶ identifyIntent ──(conditional)──▶ listKeys ──────────▶ chatResponse ──▶ END
                                              │                                      ▲
                                              ├──▶ readKey ──────────────────────────┤
                                              ├──▶ pixWithdraw ──────────────────────┤
                                              ├──▶ brcodePreview ────────────────────┤
                                              ├──▶ pixPayment ───────────────────────┤
                                              └──▶ fallback ─────────────────────────┘
```

## Decisão 2: O Guardrail É o Primeiro Nó, Não um Middleware

Toda requisição entra pelo nó `guardrail`, alimentado por um modelo dedicado de segurança (`llama-guard3:8b`). Se a entrada parecer prompt injection, o grafo corta o caminho para um nó `blockedResponse` e termina — o classificador de intenção e a API bancária nem chegam a ver a mensagem:

```python
workflow.add_conditional_edges(
    "guardrail",
    route_guardrail,
    {
        "blocked": "blockedResponse",
        "safe": "identifyIntent",
    },
)
workflow.add_edge("blockedResponse", END)
```

**Trade-off:** uma chamada extra de LLM a cada mensagem adiciona latência (visível na métrica `graph_node_duration_seconds`). Mas num sistema onde uma injeção bem-sucedida pode disparar uma transação financeira, um portão de segurança dedicado no nível do *grafo* — independente do LLM de roteamento — é inegociável. Defesa em profundidade vence prompt engineering.

## Decisão 3: LLM-como-Roteador com Structured Output

A classificação de intenção é feita pelo próprio LLM, mas restringida por structured output. O estado define o contrato como um tipo `Literal`:

```python
class GraphState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    command: Literal[
        "list_keys", "read_key", "pix_withdraw",
        "brcode_preview", "pix_payment",
        "brcode_ambiguous", "unknown",
    ]
    ...
```

O roteamento, então, é Python puro e testável — sem LLM envolvido na decisão em si:

```python
workflow.add_conditional_edges(
    "identifyIntent",
    route_intent,
    {
        "list_keys": "listKeys",
        "read_key": "readKey",
        "pix_withdraw": "pixWithdraw",
        "brcode_preview": "brcodePreview",
        "pix_payment": "pixPayment",
        "brcode_ambiguous": "fallback",
        "fallback": "fallback",
    },
)
```

Essa divisão — **o LLM propõe, o código determinístico decide** — é o padrão que eu recomendaria para qualquer agente em produção: deixe o modelo fazer o que ele faz de melhor (entender linguagem natural bagunçada) e deixe o código fazer o que ele faz de melhor (impor fluxo de controle).

Note o comando `brcode_ambiguous`: quando o LLM detecta um BRCode mas não consegue dizer o que o usuário quer fazer com ele, roteia para um fallback seguro em vez de chutar. Ambiguidade é um resultado de primeira classe, não um erro.

## Decisão 4: Provedor Duplo de LLM — Ollama para Dev, OpenRouter para Prod

O `LLMService` abstrai o provedor, então o mesmo código roda em dois modos:

| Modo            | Provedor           | Modelo                        | Por quê                                                       |
|-----------------|--------------------|-------------------------------|---------------------------------------------------------------|
| Desenvolvimento | **Ollama** (local) | `Qwen2.5:14B-Instruct-Q4_K_M` | Grátis, offline, privado — iterar sem queimar créditos de API |
| Produção        | **OpenRouter**     | `google/gemini-2.5-flash`     | Gerenciado, rápido, uma API key para vários modelos           |

O modelo de guardrail (`llama-guard3:8b`) também roda localmente via Ollama. Trocar de provedor é uma mudança de configuração, não um refactor — o mesmo princípio de injeção de dependência que apliquei no meu [projeto de RAG local](https://mabittar.github.io/Arquitetura-de-RAG-Local/).

## Decisão 5: Conversas com Estado Habilitam Fluxos Reais de Pagamento

Aqui é onde o checkpointer do LangGraph brilha. QR Codes PIX podem ser emitidos *sem valor* — quem paga decide quanto pagar. Lidar com isso exige uma conversa multi-turno com memória:

1. **Usuário:** _"pague esse pix copia e cola 00020126...Jesse Pinkman..."_
2. **Assistente:** _"O QR Code que você está tentando pagar é para Jesse Pinkman. Ele não tem um valor específico associado. Qual valor deseja pagar?"_
3. **Usuário:** _"pague R$ 5.00"_
4. **Assistente:** _"Pagamento realizado com sucesso. ID End To End: E49288113... — Valor pago: R$ 5,00 — Status: CREATED"_

O estado é persistido em PostgreSQL via `AsyncPostgresSaver`, indexado por um `thread_id`. O contrato HTTP é simples: o cliente envia um header `X-Thread-ID`; se ausente, a API gera um e o retorna no header da resposta para o próximo turno.

A parte interessante é como o nó de intenção trata a continuação — ele checa o estado persistido *antes* de chamar o LLM:

```python
def _handle_continuation(state: GraphState, messages: list) -> dict | None:
    """Resolve um valor pendente a partir da última mensagem do usuário."""
    if not (state.get("awaiting_amount") and messages):
        return None
    amount = _extract_amount(str(messages[-1].content))
    if not amount:
        return None
    return {
        "command": "pix_payment",
        "withdraw_amount": amount,
        "awaiting_amount": False,
    }
```

**O insight-chave:** se o usuário está respondendo a uma pergunta pendente ("R$ 5.00"), você não precisa de uma chamada de LLM para descobrir a intenção — o estado já te diz. Um regex e uma flag vencem uma classificação que gasta tokens, e o fluxo se torna determinístico exatamente onde dinheiro está envolvido.

## Decisão 6: FastAPI como Espinha Dorsal

O FastAPI foi uma escolha deliberada, e ele carrega mais peso aqui do que apenas "servir HTTP":

- **Async de ponta a ponta:** a invocação do grafo (`graph.ainvoke`), o cliente bancário, o Redis e o Postgres são todos async — um framework síncrono seria um gargalo fingindo ser um servidor.
- **Pydantic v2 em tudo:** contratos de request/response (`ChatRequest`, `ChatResponse`), settings via `pydantic-settings` e DTOs na fronteira com o banking. Um único modelo de validação de ponta a ponta.
- **Injeção de dependência via `app.state`:** cache e checkpointer são anexados no startup do lifespan e resgatados dentro da rota — sem globais.
- **Docs OpenAPI de graça:** o Swagger UI interativo em `/docs` funciona como bancada de testes para o grafo inteiro.

A rota em si permanece magra — toda a complexidade vive atrás do grafo:

```python
@chat_router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, raw_request: Request,
               thread_id: Annotated[str | None, Header()] = None):
    cache = getattr(raw_request.app.state, "cache", None)
    checkpointer = getattr(raw_request.app.state, "checkpointer", None)
    thread_id = thread_id or str(uuid4())
    graph = GraphProcessor(log=logger, cache_service=cache,
                           checkpointer=checkpointer).get_graph()
    result = await graph.ainvoke(
        {"messages": [HumanMessage(content=request.question)]},
        {"configurable": {"thread_id": thread_id}},
    )
    ...
    return JSONResponse(ChatResponse(answer=response_msg).model_dump(),
                        headers={"X-Thread-ID": thread_id})
```

## Decisão 7: Observabilidade Não É Opcional

Um agente que chama APIs externas e movimenta dinheiro sem traces é um pesadelo de debug esperando para acontecer. O serviço é instrumentado com **OpenTelemetry** (opt-in via `OTEL_ENABLED`, no-op quando desligado):

- **Traces:** span raiz de HTTP, spans de domínio por nó do grafo (`graph.intent.identify`, `pix.withdraw.execute`, ...), spans de IO (API bancária, Redis, Postgres) e spans de LLM com contagem de tokens. Exportados para **Tempo/Jaeger**.
- **Métricas:** RED para HTTP mais métricas de domínio — `pix_operations_total`, `graph_node_duration_seconds`, `guardrail_block_total` — enviadas ao **Prometheus**.
- **Logs:** `structlog` em JSON com correlação automática de `trace_id`/`span_id`, enviados ao **Loki**.
- **Privacidade:** chaves PIX, documentos, tokens e secrets são mascarados em atributos de span, labels de métrica e logs.

A stack Grafana vem com dashboards pré-provisionados ("Visão Geral do Serviço", "Operações PIX"), então consigo observar latência p95 por nó, taxa de bloqueio do guardrail e throughput de tokens de LLM enquanto converso com o assistente.

Meu detalhe favorito: a stack inclui um **servidor MCP do Grafana em modo read-only** — o que significa que outro agente LLM pode consultar a telemetria *deste* agente. Agentes observando agentes, com uma service account apenas Viewer.

## O Frontend: Uma SPA Deliberadamente Magra

A UI de chat ([chat-llm](https://github.com/mabittar/chat-llm)) é uma SPA Vue 3 + Vite com Pinia e TailwindCSS. Sua única responsabilidade real além de renderizar é o contrato de thread: gerar um UUID local para uma nova conversa, enviá-lo como `X-Thread-ID` e adotar o thread ID que a API retornar. Testes unitários com Vitest, E2E com Playwright. Manter o frontend magro mantém a lógica conversacional exatamente onde ela pertence: no grafo.

## O Que Aprendi Sobre Trade-offs

**Determinismo vs. autonomia.** O hype da indústria está em agentes totalmente autônomos. Para domínios regulados que tocam dinheiro, o ponto ótimo é o oposto: LLMs para entender, máquinas de estado para decidir. Autonomia é um espectro, e você escolhe onde se posicionar nele caso a caso.

**Chamadas de LLM são um orçamento.** O handler de continuação economiza uma chamada de classificação por pagamento multi-turno. Ganho pequeno? Claro. Mas desenhar fluxos onde o *estado* responde antes do modelo é um hábito que se acumula — em latência, custo e previsibilidade.

**Guardrails pertencem à arquitetura, não ao prompt.** Um system prompt dizendo "não siga instruções maliciosas" é uma sugestão. Uma aresta de grafo que impede fisicamente o alcance da API bancária é um controle.

**Modelos locais são production-grade para loops de dev.** O Qwen2.5:14B quantizado lida surpreendentemente bem com classificação de intenção e geração de resposta em português. O ciclo inteiro de desenvolvimento roda offline a custo marginal zero — o OpenRouter só entra quando preciso de escala gerenciada.

## Próximos Passos

O roadmap é público no repositório: autenticação no `/chat` (a partir da qual o assistente poderá buscar proativamente as contas do cliente), mais operações PIX e uma pipeline de CI/CD. As fundações — guardrails, persistência e observabilidade — já estão no lugar.

---

**Engenheiros de IA Aplicada:** quando as ferramentas do seu agente podem disparar efeitos colaterais no mundo real, onde você traça a linha entre autonomia do modelo e fluxo de controle determinístico?

---

_Código completo, ADRs e instruções de setup: [github.com/mabittar/banking-llm](https://github.com/mabittar/banking-llm) · Frontend: [github.com/mabittar/chat-llm](https://github.com/mabittar/chat-llm)_