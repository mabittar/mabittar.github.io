---
layout: post
title: Arquitetura de RAG Local - Como Eliminei a Dependência de Vector DBs em Nuvem
subtitle: FastAPI + pgvector + OpenRouter - Trade-offs de uma arquitetura híbrida para IA aplicada
date: 2026-04-21 00:00:00 +0300
description: Arquitetura de RAG local com FastAPI, PostgreSQL pgvector e OpenRouter. Trade-offs entre custo, privacidade e complexidade na implementação de pipelines de IA.
img: rag_architecture.png
tags: [RAG, FastAPI, PostgreSQL, pgvector, OpenRouter, Arquitetura de Software, IA Aplicada, LLM]
---

## O Problema Silencioso dos Pipelines RAG Modernos

A maioria das equipes está pagando valores significativos por soluções de RAG gerenciadas em nuvem quando poderiam ter total controle sobre seu pipeline por uma fração do custo. Mas essa decisão envolve trade-offs que vão muito além do preço.

Nos últimos meses, venho arquitetando uma plataforma de RAG local que desafia o status quo de usar múltiplos serviços em nuvem. O resultado é um sistema que mantém dados sensíveis localmente, reduz custos operacionais e oferece tempos de resposta comparáveis às soluções gerenciadas.

## A Arquitetura Híbrida: Por Que Não Usei ChromaDB ou Pinecone?

### A Decisão do Banco de Vetores

A escolha mais controversa foi eliminar um banco de dados vetorial dedicado. Em vez de ChromaDB, Pinecone ou Weaviate, optei por **PostgreSQL com a extensão pgvector**.

**Trade-off analisado:**

| Solução | Prós | Contras | Decisão |
|---------|------|---------|---------|
| **PostgreSQL + pgvector** | ACID compliance, JOINs com metadados, backups unificados, HNSW index | Setup inicial mais complexo | ✅ Escolhido |
| ChromaDB | Zero-config, otimizado para embeddings | Dual database (vetores + metadados separados), menos escalável | ❌ |
| Pinecone | Gerenciado, auto-scaling | Lock-in de vendor, custo, dados saem da infraestrutura | ❌ |
| FAISS | Extremamente rápido, GPU support | Apenas índice, sem metadados nativos | ❌ |

**O insight chave:** PostgreSQL com pgvector elimina a necessidade de manter dois bancos de dados. A fonte única de verdade simplifica backups, garante consistência transacional e permite consultas híbridas (semântica + filtragem SQL) em uma única query.

### Configuração do Índice HNSW

A configuração do índice HNSW (Hierarchical Navigable Small World) foi crítica para performance:

- **M = 16**: Número de conexões por elemento (trade-off entre memória e precisão)
- **ef_construction = 64**: Fator de busca durante construção (maior = melhor qualidade, mais lento para build)
- **ef_search = 100**: Fator de busca durante queries (ajustável para precision vs. velocidade)

Com 384 dimensões (all-MiniLM-L6-v2), a busca por similaridade com operador L2 distance mantém-se consistentemente abaixo de 5ms para collections de até 100k chunks.

## Estratégia Dual de LLM: OpenRouter + Ollama

### O Dilema Local vs. Nuvem

A arquitetura suporta dois modos de operação, cada um com seu perfil de trade-offs:

**Modo POC (OpenRouter):**
- Modelos free tier: Llama 3.2, Mistral, DeepSeek
- Sem necessidade de GPU local
- Rate limits aplicáveis (mas generosos para free tier)
- Requer conexão com internet

**Modo em Escala:**
- 100% offline, privacidade total
- Requer GPU dedicada e setup complexo
- Sem rate limits

A decisão arquitetural foi isolar o provider LLM através de uma interface unificada. O service layer utiliza injeção de dependência para alternar entre providers sem modificar a lógica de negócio.

### Streaming via Server-Sent Events (SSE)

A escolha de SSE sobre WebSockets para streaming de respostas LLM:

**Por que SSE venceu:**
- Unidirecional é suficiente (servidor → cliente apenas)
- Suporte nativo do browser via EventSource
- Reconexão automática
- Menor complexidade que WebSockets

**Trade-off:** Cliente não pode enviar mensagens pela mesma conexão, mas para chat isso é aceitável (HTTP POST para input, SSE para output).

## Modularização e Separation of Concerns

A arquitetura segue um **Monolito Modular** com camadas bem definidas:

```
Presentation Layer (Vue.js 3 - Dark Mode)
    ↓
API Layer (FastAPI Routers)
    ↓
Service Layer (RAG, Document, Chat Services)
    ↓
Infrastructure Layer (OpenRouter Client, PGVector Store)
    ↓
Data Layer (PostgreSQL + pgvector)
```

**Princípio fundamental:** O Service Layer é completamente desacoplado de providers externos. Trocar OpenRouter por Ollama (ou adicionar OpenAI, Anthropic) requer apenas implementar a interface base, sem tocar em RAGService ou ChatService.

### Estratégia de Chunking

O chunking inteligente é onde muitos pipelines RAG falham:

- **Tamanho:** 512 tokens (equilíbrio entre granularidade e contexto)
- **Overlap:** 50 tokens (mantém continuidade semântica entre chunks)
- **Estratégia:** RecursiveCharacterTextSplitter com separadores hierárquicos (parágrafos → sentenças → palavras)

**Trade-off:** Overlap maior melhora recall mas aumenta storage. Com pgvector unificado, o overhead é gerenciável.

## Escalabilidade Horizontal: O Que Falta

A arquitetura atual é intencionalmente um monolito. Para escalar horizontalmente, seriam necessários:

1. **Redis/Celery** para fila de processamento de documentos (hoje síncrono)
2. **Read replicas** do PostgreSQL para queries de vetor
3. **CDN** para assets do frontend (hoje servido via Vite dev server)
4. **Load balancer** com sticky sessions para SSE

**Decisão consciente:** Não implementar esses componentes na POC. Cada um adiciona complexidade operacional que só se justifica após validação do product-market fit.

## Casos de Uso Além do Document Q&A

Esta arquitetura não se limita a chat com documentos. Ela escala para:

- **Bases de conhecimento interno:** HR policies, documentação de engenharia, playbooks
- **Automação de suporte ao cliente:** Com histórico de conversas persistido
- **Trilhas de auditoria para compliance:** Cada query logada com suas fontes (font attribution)
- **Sistemas de recomendação:** Similaridade entre documentos e perfis de usuário

## O Que Aprendi Sobre Trade-offs

**Custo vs. Complexidade:** Soluções gerenciadas abstraem complexidade por um preço. Quando você precisa de controle total (compliance, privacidade), a complexidade operacional se torna um investimento, não um custo.

**Escolha de Banco de Dados:** Unificar dados estruturados e vetoriais em PostgreSQL simplifica operações mas limita algumas otimizações de bancos especializados. Para 99% dos casos de uso corporativo, a diferença de performance é negligenciável comparada à simplicidade operacional.

**Modelos Locais vs. API:** Modelos locais (Ollama) oferecem privacidade total mas requerem hardware especializado. APIs externas (OpenRouter) democratizam acesso mas criam dependência. A arquitetura dual permite migração gradual conforme os requisitos evoluem.

## Próximos Passos

O roadmap inclui:

1. **Implementar MarkItDown** com fallback para parsers tradicionais em caso de erro
2. **MVP LangGraph** - migrar apenas o pipeline de query inicialmente
3. **Coletar dados de treinamento** para fine-tuning (Q&A de usuários reais)


---

**Engenheiros de IA aplicada:** Qual é sua maior preocupação ao migrar de serviços gerenciados de IA para pipelines self-hosted — controle, custo ou complexidade operacional?

---

*Para ver o código completo e a documentação técnica detalhada, acesse o repositório do projeto em [github.com/mabittar/local-rag](https://github.com/mabittar/local-rag)*
