---
layout: post
title: "O Harness Científico de Auto-Evolução: Medindo Sistemas Agenticos Sem Vibe Coding"
subtitle: "Como transformar hipóteses em fatos verificáveis através de loops populacionais, verificação física e ROI matemático"
date: 2026-09-01 08:00:00 +0300
description: "Harness científico para auto-evolução de agentes — abandon vibe coding e aderir a medição rigorosa, traços de execução e ROI de mutações."
img: harness_evolution.jpg
tags:
  [
    Agentic AI,
    Harness Design,
    Test-Time Reinforcement Learning,
    Scientific Method,
    Performance Evaluation,
    Claude Code,
    Hermes Agent,
  ]
---

## A Ilusão do "Vibe Coding" em Sistemas Agenticos

Conversava recentemente com um colega sobre o futuro de sistemas agenticos auto-evolutivos. E uma questão brilhante surgiu:

> _Como você constrói um agente capaz de evoluir, equipado com rigor científico, que possa transformar hipóteses em fatos verificáveis? Que analise sistematicamente, realize medições, comparações, e separe o que realmente é melhoria do que é apenas ruído?_

Hoje, temos sistemas agenticos sofisticados. Eles não apenas geram texto — eles _agem_. Interagem com terminais, repositórios, APIs. Tomam decisões sob incerteza. E cada decisão é uma hipótese que precisa ser testada e validada.

Mas aqui está o problema:

Quando um engenheiro senior afirma _"essa mudança no prompt melhorou a qualidade"_, o que ele realmente mediu?

- Sensação pessoal?
- Um benchmark isolado (cherry-picked)?
- Uma comparação com "gold labels" (que sempre favorece mudanças recentes)?

**Isto é "vibe coding"** — e é o inimigo silencioso da engenharia agentica.

---

## A Revelação: 1,6% IA + 98,4% Harness

Aqui vem o insight que muda tudo.

Análises arquiteturais de sistemas agenticos modernos revelam um padrão consistente:

- **1,6% do código:** Lógica de decisão baseada em IA (pesos do LLM, semântica)
- **98,4% do código:** Infraestrutura operacional do harness (orquestração, state management, recuperação de falhas, verificação física)

Isto não é um detalhe. É _tudo_.

Porque significa:

```
Um bom harness com um LLM mais fraco
  >
Um péssimo harness com um LLM mais forte
```

Você pode congelar os pesos de um modelo e simplesmente **mutar o harness que o envolve**, e observar diferenças empíricas de até **18 pontos em benchmarks de horizonte longo** e **6x no desempenho geral**. Isto é fato medido, não opinião.

**Os pesos do modelo não evoluem durante operação. O harness evolui.**

Portanto, a pergunta errada é: _"Qual é o melhor modelo (GPT? Claude? Grok)?"_

A pergunta certa é: _"Qual é o harness que extrai máximo potencial do modelo que temos?"_

---

## Arquitetura do Agente Científico: Três Papéis, Um Loop

Para construir um harness de auto-evolução **sem cair em vibe coding**, precisamos de uma arquitetura baseada em **evidência reproduzível**.

Não é um enxame caótico (swarm). Não é um grafo rígido (LangGraph). É um **loop populacional guiado por ROI**.

### **Papel 1: SOLVER — O Executor Físico**

```
Responsabilidade: Executar a hipótese (teste) contra o sistema alvo
                 e coletar EVIDÊNCIA DETERMINÍSTICA e REPRODUZÍVEL
```

O Solver não interpreta. Não opina. Apenas executa e registra. Transforma experimento em fato.

**Saídas obrigatórias:**

| Output                      | Descrição                                                  | Por quê?                                      |
| --------------------------- | ---------------------------------------------------------- | --------------------------------------------- |
| **Execution trace**         | Sequência de chamadas de ferramentas, tempos, stack traces | Permite replicação exata                      |
| **Verification Story**      | "Arquivos foram criados? Contratos de código mantidos?"    | Testa se o agente _realmente_ fez o que disse |
| **Stderr/Stdout**           | Logs brutos, parseados para sinais de erro                 | Separa falhas reais de falsos positivos       |
| **Latency (P50, P95, P99)** | Distribuição de tempo, não apenas média                    | Identifica cauda longa de latência            |
| **Token consumption**       | Prompt + completion, por step                              | Detecta vazamentos de contexto                |
| **Success/Failure signal**  | Binário, baseado em verificação física                     | Sem semântica, sem interpretação              |

**Pseudocódigo ilustrativo:**

```python
def solver_execute(harness_version: str, benchmark_case: str):
    """Executa aplicação, coleta evidência física reproduzível"""

    with isolated_environment(seed=42):  # Seed para reprodutibilidade
        app = load_app_with_harness(harness_version)

        tracer = Tracer()
        start = perf_counter()

        try:
            result = app.run(benchmark_case)
            # Verificação FÍSICA, não semântica
            success = verify_contracts(result) and \
                      check_filesystem_changes() and \
                      validate_json_schema(result)
        except Exception as e:
            success = False

        elapsed = perf_counter() - start

        return {
            "success": success,
            "trace": tracer.dumps(),
            "latency_ms": elapsed * 1000,
            "tokens_used": tracer.count(),
            "verification_story": {
                "files_created": list_new_files(),
                "contracts_met": success,
                "state_transitions": tracer.state_deltas()
            }
        }
```

### **Papel 2: PROPOSER — O Cientista Diagnosticador**

```
Responsabilidade: Ler falhas do Solver, propor MUTAÇÕES DETERMINÍSTICAS
                 no harness (nunca nos pesos, nunca em prompt magic)
```

O Proposer é um "cientista" que examina o traço de execução e formula hipóteses testáveis.

**Mutações candidatas típicas:**

| Tipo                    | Mudança                                                  | Quando Aplicar                  |
| ----------------------- | -------------------------------------------------------- | ------------------------------- |
| **Context Strategy**    | Agresiva compactação de metadados, lazy-loading de tools | Tokens usados > 85% do limite   |
| **Retry Logic**         | Backoff adaptativo vs. exponencial vs. fail-fast         | Retry count > 3 em traços       |
| **Tool Ranking**        | Reordenar tools por frequência histórica de sucesso      | Taxa de sucesso de tool < 70%   |
| **State Checkpointing** | Interval de persistência de estado (a cada N steps?)     | Latência de recuperação crítica |
| **Error Recovery**      | Graceful degrade vs. exception propagation               | Falhas específicas recorrentes  |
| **Output Truncation**   | Limitar output de ferramentas (1000 vs. 2000 tokens)     | Token overhead em tool calls    |

**Exemplo real de diagnóstico:**

Você está analisando traços de execução de um sistema agentico. Observa:

- P95 latência subiu 40% nos últimos 7 dias
- Retry count aumentou de 1.2 para 3.8 por sessão
- Tokens por chamada cresceram 22%

**Diagnóstico:**

```
Hipótese: Tool output está inflado, causando retry loops.
Mutação proposta:
  - Reduzir tool_output_max_tokens de 2000 → 1000
  - Implementar tool output summarization

Rationale: Traço mostra 19% de latência vem de retry storms.
          Se output é truncado, primeira tentativa tem mais contexto útil.

Risco: Tool output pode perder informação crítica.
Trade-off: Latência vs. fidelidade de informação.
```

### **Papel 3: JUDGE — O Árbitro (Sem Gold Labels)**

```
Responsabilidade: Selecionar MELHOR mutação usando SINAIS FÍSICOS APENAS
                 Proibido comparar com "resposta correta"
```

Este é o papel mais crítico. Porque aqui você **não pode trapaçear com viés semântico**.

O Judge vê apenas:

- ✅ Latência (melhorou?)
- ✅ Taxa de sucesso (erros diminuíram?)
- ✅ Token efficiency (consumo caiu?)
- ✅ ROI (custo da mudança < ganho?)

O Judge **nunca** vê:

- ❌ "A resposta parece mais bonita"
- ❌ "Tenho uma boa sensação sobre isto"
- ❌ "Semanticamente, ficou melhor"

**Pseudocódigo do Judge:**

```python
def judge_select_best_mutation(mutations: List[Mutation],
                               baseline: SolverOutput,
                               new_outputs: Dict[Mutation, SolverOutput]) -> Mutation:
    """Seleciona mutação vencedora usando APENAS sinais físicos"""

    scores = {}

    for mutation, output in new_outputs.items():
        # Critério 1: Latency (P95)
        latency_gain = (baseline.latency_p95 - output.latency_p95) / baseline.latency_p95
        latency_score = max(0, latency_gain)  # 0 se piorou

        # Critério 2: Success rate
        baseline_success = baseline.verification_story["contracts_met"]
        output_success = output.verification_story["contracts_met"]
        success_delta = output_success - baseline_success
        success_score = 1.0 if success_delta >= 0 else -0.5

        # Critério 3: Token efficiency
        token_gain = (baseline.tokens_used - output.tokens_used) / baseline.tokens_used
        token_score = max(0, token_gain)

        # Critério 4: ROI (ganho vs. custo da mudança)
        mutation_cost = estimate_operational_cost(mutation)
        total_gain = (latency_score * 0.4 +
                      success_score * 0.3 +
                      token_score * 0.3)
        roi = total_gain - mutation_cost

        scores[mutation] = roi

    # Retornar mutação com maior ROI
    # NÃO é "a que melhor soou", é "a que melhor performou"
    winner = max(scores.items(), key=lambda x: x[1])[0]
    return winner
```

---

## Critérios Que Não Podem Faltar

### **1. Verificação por Ferramentas Físicas (T³RL)**

Não é suficiente:

```
"O modelo gerou uma resposta"
```

É necessário:

```
"A ferramenta foi invocada → retornou JSON válido →
 foi parseada → afetou estado do filesystem →
 transação confirmada → verificação de integridade passou"
```

**Implementação:**

```python
class PhysicalVerification:
    def verify_tool_execution(self, tool_call: ToolCall, result: Any) -> bool:
        """Valida que a ferramenta realmente executou com consequência"""

        checks = {
            "json_valid": is_valid_json(result),
            "schema_matches": matches_expected_schema(result),
            "has_side_effects": (
                check_filesystem_changes() or
                check_database_changes() or
                check_api_calls_logged()
            ),
            "is_idempotent": can_replay_safely(tool_call),
            "contracts_met": verify_preconditions_postconditions()
        }

        # TUDO deve passar, não "maioria"
        return all(checks.values())
```

### **2. Análise de Traços (Execution Traces)**

Não basta a resposta final. O caminho importa.

**Componentes de um trace útil:**

```
Tool call sequence
  ├─ Qual ordem? (reflete prioridades do modelo)
  └─ Callbacks corretos? (state propagated?)

Token usage per step
  ├─ Onde está o overhead?
  └─ Qual ferramenta consome mais?

Error recovery events
  ├─ Quantas falhas?
  ├─ Quais estratégias funcionaram?
  └─ Qual é a taxa de sucesso por tipo?

State mutations
  ├─ Memória alterada?
  ├─ Arquivos criados?
  └─ Transações commitadas?

Latency breakdown
  ├─ LLM inference: X ms
  ├─ Tool execution: Y ms
  ├─ Parsing: Z ms
  └─ Retry loops: W ms (onde podemos melhorar?)
```

**Exemplo real:**

```
Trace de falha em Claude Code v1.4:

Step 1: user_query → encoding           12ms  ✓
Step 2: tool_selection (LLM)            450ms ✓
Step 3: bash_execution                  5ms   ✗ [permission denied]
Step 4: error_recovery → retry+sudo     200ms ⚠️ [3 attempts]
Step 5: final_result (generation)       380ms ✓

Latência total: 1047ms
% causado por retry: 200/1047 = 19% (reduzível!)

AÇÃO: Pré-validar permissões antes de executar
      Proposta: verificar `stat` antes de `rm`
```

### **3. Equação de Viabilidade (Custo Total)**

Nem toda mutação vale a pena evoluir.

```
C_total = N_variants × N_traces × (C_prompt + C_generation) + C_judge

Onde:
  N_variants = quantas mutações testar
  N_traces = quantos benchmarks rodar para cada
  C_prompt = tokens de input (análise Proposer)
  C_generation = tokens de output (propostas)
  C_judge = tokens da seleção final
```

**Exemplo numérico (análise Hermes Agent):**

```
N_variants: 5 mutações
N_traces: 20 casos de teste
C_prompt: 2000 tokens/trace
C_generation: 500 tokens/trace
C_judge: 1000 tokens
Preço: $0.0075 por 1M tokens

C_total = 5 × 20 × (2000 + 500) / 1M × $0.0075
        + 1000 / 1M × $0.0075
        = 100 × 0.01875 + 0.0000075
        ≈ $0.38 por geração

Ganho esperado:
  - Latência reduzida 15% → 1000 chamadas/dia = 150s poupados
  - Tokens reduzidos 12% → 50k tokens/dia = 6k tokens poupados
  - Valor por token economizado (em compute): ~$0.000001

  ROI ≈ (150s × micro_latency_value + 6k × $0.000001) / $0.38

Viável? Sim, se o ganho composto > 1000x custo.
```

### **4. Contexto como Gargalo (Lazy-Loading)**

O tamanho do contexto é o recurso mais escasso em sistemas agenticos.

**Camadas de compactação (observadas em Claude Code real):**

```
1. Metadata striping
   └─ Remover IDs desnecessários, timestamps não críticos

2. Tool schema on-demand
   └─ Carregar `tool.json` SÓ quando a ferramenta é selecionada

3. Error alert aggregation
   └─ Resumir "erro N" idênticos em 1 item de log

4. Microcompression
   └─ Tokenização otimizada para estruturas de dados conhecidas

5. LLM-based summarization
   └─ Última resort — subagente resume conteúdo anterior
```

**Métrica crítica:**

```
Contexto total disponível:      100k tokens (Claude 3.5 Sonnet)
Contexto utilizado por sessão:  89k tokens
Margem de segurança:            11% (CRÍTICO!)

Ação necessária:
  ├─ Implementar lazy-loading de tool schemas
  ├─ Threshold: carregar tool X SÓ se foi usada em últimas 3 turns
  └─ Resultado esperado: reduzir para 75k tokens, margem = 25% (confortável)
```

---

## Arquitetura: Loop Populacional vs. Swarm vs. Grafo

Qual é melhor? **Nenhuma das três puras.**

### **Padrão Ideal: Population-Based Loop com Selective Breeding**

```
Iteração N:
  População atual: [Harness_v1, Harness_v2, Harness_v3]

  ├─ Solver executa cada versão (3 × 20 traces) → 60 SolverOutputs
  ├─ Proposer analisa falhas comuns → 5-7 mutações candidatas
  ├─ Judge seleciona top 3 mutações por ROI
  └─ Recombinar: cruzar características (winner + runner-up)

  ↓

  Iteração N+1: [Harness_winner, Harness_hybrid1, Harness_hybrid2]

Critério de parada:
  - Nenhuma mutação gera ROI > 1.1x em 5 gerações consecutivas
  - OU C_total acumulado > orçamento de evolução
  - OU algoritmo converge em plateau (lei dos rendimentos decrescentes)
```

**Por que não Swarm?**

- Swarms divergem caoticamente, sem direção inequívoca
- Impossível replicar resultado (stochástico demais)
- Difícil provar que uma mudança "realmente ajudou"

**Por que não LangGraph puro?**

- Grafos rígidos previnem descoberta de padrões novos
- Cada caminho é hardcodado; evolução é limitada a thresholds

**Por que Population-Based Loop?**

- ✅ Determinístico (reproducible com seed)
- ✅ Converge em direção inequívoca (ROI matemático)
- ✅ Prova científica clara: _"Mutação X venceu 12% em latência"_
- ✅ Permite recombinação (evolução genética, mas com rigor)

---

## Pesos do Modelo vs. Harness: A Realidade Matemática

Você tem um LLM com N bilhões de parâmetros. Quanto realmente ele importa?

```
Desempenho Final = f(Pesos_LLM, Harness)

Empiricamente (Claude Code + análise de código fonte):
  - Pesos do LLM: 99.8% fixos durante operação (congelados)
  - Harness: 98.4% da superfície de controle (mutável)

∴ Impacto de mudanças:
  - Alterar pesos (fine-tuning): ±2-5 pontos em benchmarks
  - Alterar harness (só context mgmt): ±18 pontos (9x maior!)
  - Alterar harness (retry + tool ranking): 6x performance geral
```

**O que isto significa:**

```
LLM é o "pensador"    — decide qual ferramenta usar
Harness é a "estrutura nervosa" — decide:

  ├─ Quantas chances o LLM tem de acertar (retry logic)
  ├─ Qual contexto o LLM enxerga (compactação)
  ├─ Em qual ordem o LLM escolhe (tool ranking)
  ├─ Como erros são recuperados (error strategy)
  └─ Quando o LLM deve parar (exit conditions)

Um LLM fraco com harness excelente > Um LLM forte com harness ruim
```

---

## Aplicando a Prática: Análise de Hermes Agent

Suponha que você quer avaliar **Hermes Agent** usando este framework.

### **Fase 1: Baseline (Solver)**

Você executa Hermes contra um benchmark de 20 casos:

- Resolver problemas de código
- Debugar erros
- Otimizar performance

Resultado: Hermes obtém 72% de sucesso, latência média 3.2s, token médio 45k.

### **Fase 2: Diagnóstico (Proposer)**

Você examina os traços de falha dos 28% que falharam:

```
Padrão 1: Tool selection ruim
  └─ Hermes escolhe "grep" quando deveria usar "find"
     Taxa de sucesso de "grep": 61%
     Taxa de sucesso de "find": 89%

Padrão 2: Context overflow
  └─ Após 5 iterações, contexto satura
     Último step: 98k tokens usados, erro de truncação

Padrão 3: Retry storms
  └─ Quando "bash execution" falha, Hermes tenta 7x em vez de 3x
     Latência P95: 8.5s (causada por retry loops)
```

Proposer gera 5 mutações:

1. **Tool ranking:** Reordenar tools por taxa histórica de sucesso
2. **Context strategy:** Implementar lazy-loading de tool schemas
3. **Retry logic:** Limitar retries a 3, com backoff adaptativo
4. **Output truncation:** Reduzir tool_output_max_tokens de 3000 → 1500
5. **State checkpointing:** Persister estado a cada 4 steps

### **Fase 3: Seleção (Judge)**

Você executa cada mutação contra o mesmo benchmark:

| Mutação             | Sucesso | Latência | Tokens | ROI              |
| ------------------- | ------- | -------- | ------ | ---------------- |
| Baseline            | 72%     | 3.2s     | 45k    | —                |
| Tool ranking        | 78%     | 3.0s     | 44k    | **+8.3%** ✅     |
| Context strategy    | 72%     | 3.1s     | 38k    | +6.2%            |
| Retry logic         | 74%     | 2.5s     | 42k    | +12.1% ✅        |
| Output truncation   | 68%     | 3.0s     | 40k    | -2.3% (rejeitar) |
| State checkpointing | 73%     | 3.4s     | 45k    | +1.5%            |

**Winner:** Retry logic (+12.1% ROI)

### **Fase 4: Próxima Geração**

Você combina:

- Retry logic (winner) +
- Tool ranking (runner-up) +
- Partial context strategy (promissora)

Nova população: [Hermes_winner, Hermes_hybrid1, Hermes_hybrid2]

Repete ciclo até convergência.

---

## Trade-Offs Críticos

| Aspecto             | High Precision      | High Speed      | Balanço Recomendado |
| ------------------- | ------------------- | --------------- | ------------------- |
| **N_traces**        | 100+ casos          | 10 casos        | 20-30 (sweet spot)  |
| **Retry strategy**  | Exponential backoff | Fail-fast       | Adaptive jitter     |
| **Tool output**     | 5000 tokens         | 500 tokens      | 1500 tokens         |
| **Checkpoint freq** | A cada step         | A cada 10 steps | A cada 4 steps      |
| **Evolution iters** | 50+ gerações        | 3 gerações      | 10-15 gerações      |

Você **não pode otimizar tudo**. Escolha 2 variáveis críticas para seu caso.

---

## Próximos Passos

1. **Implementar Solver:** Criar função que executa agente e coleta traços
2. **Implementar Proposer:** Parser de falhas, gerador de mutações determinísticas
3. **Implementar Judge:** Seletor por ROI (não por vibes)
4. **Validar em Hermes Agent:** Rodar ciclo completo, documentar resultados
5. **Escalar para Claude Code:** Aplicar framework ao seu próprio harness

---

**Para engenheiros de sistemas agenticos:** Qual variável você otimizaria primeiro — latência, sucesso, ou eficiência de tokens? E como você mediria "melhoria real" sem cair em vibe coding?

---

_Próximo post: Implementação prática de T³RL (Test-Time Reinforcement Learning) com verificação de ferramentas físicas._
