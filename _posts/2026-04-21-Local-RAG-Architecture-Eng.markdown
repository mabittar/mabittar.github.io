---
layout: post
title: Local RAG Architecture - How I Eliminated Dependency on Cloud Vector DBs
subtitle: FastAPI + pgvector + OpenRouter - Trade-offs of a hybrid architecture for applied AI
date: 2026-04-21 00:00:00 +0300
description: Local RAG architecture with FastAPI, PostgreSQL pgvector, and OpenRouter. Trade-offs between cost, privacy, and complexity in implementing AI pipelines.
img: rag_architecture.png
tags: [RAG, FastAPI, PostgreSQL, pgvector, OpenRouter, Software Architecture, Applied AI, LLM]
---

## The Silent Problem of Modern RAG Pipelines

Most teams are paying significant amounts for cloud-managed RAG solutions when they could have total control over their pipeline for a fraction of the cost. But this decision involves trade-offs that go far beyond price.

In recent months, I've been architecting a local RAG platform that challenges the status quo of using multiple cloud services. The result is a system that keeps sensitive data locally, reduces operational costs, and offers response times comparable to managed solutions.

## Hybrid Architecture: Why I Didn't Use ChromaDB or Pinecone?

### The Vector Database Decision

The most controversial choice was eliminating a dedicated vector database. Instead of ChromaDB, Pinecone, or Weaviate, I opted for **PostgreSQL with the pgvector extension**.

**Trade-off analysis:**

| Solution | Pros | Cons | Decision |
|---------|------|---------|---------|
| **PostgreSQL + pgvector** | ACID compliance, JOINs with metadata, unified backups, HNSW index | More complex initial setup | ✅ Chosen |
| ChromaDB | Zero-config, optimized for embeddings | Dual database (separate vectors + metadata), less scalable | ❌ |
| Pinecone | Managed, auto-scaling | Vendor lock-in, cost, data leaves infrastructure | ❌ |
| FAISS | Extremely fast, GPU support | Index only, no native metadata | ❌ |

**The key insight:** PostgreSQL with pgvector eliminates the need to maintain two databases. A single source of truth simplifies backups, ensures transactional consistency, and allows hybrid queries (semantic + SQL filtering) in a single query.

### HNSW Index Configuration

The HNSW (Hierarchical Navigable Small World) index configuration was critical for performance:

- **M = 16**: Number of connections per element (trade-off between memory and precision)
- **ef_construction = 64**: Search factor during construction (higher = better quality, slower build)
- **ef_search = 100**: Search factor during queries (adjustable for precision vs. speed)

With 384 dimensions (all-MiniLM-L6-v2), the similarity search with L2 distance operator consistently stays below 5ms for collections up to 100k chunks.

## Dual LLM Strategy: OpenRouter + Ollama

### The Local vs. Cloud Dilemma

The architecture supports two modes of operation, each with its own trade-off profile:

**POC Mode (OpenRouter):**
- Free tier models: Llama 3.2, Mistral, DeepSeek
- No local GPU required
- Applicable rate limits (but generous for free tier)
- Requires internet connection

**Scale Mode:**
- 100% offline, total privacy
- Requires dedicated GPU and complex setup
- No rate limits

The architectural decision was to isolate the LLM provider through a unified interface. The service layer uses dependency injection to switch between providers without modifying business logic.

### Streaming via Server-Sent Events (SSE)

Choosing SSE over WebSockets for streaming LLM responses:

**Why SSE won:**
- Unidirectional is enough (server → client only)
- Native browser support via EventSource
- Automatic reconnection
- Lower complexity than WebSockets

**Trade-off:** Client cannot send messages over the same connection, but for chat this is acceptable (HTTP POST for input, SSE for output).

## Modularization and Separation of Concerns

The architecture follows a **Modular Monolith** with well-defined layers:

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

**Fundamental principle:** The Service Layer is completely decoupled from external providers. Swapping OpenRouter for Ollama (or adding OpenAI, Anthropic) only requires implementing the base interface, without touching RAGService or ChatService.

### Chunking Strategy

Smart chunking is where many RAG pipelines fail:

- **Size:** 512 tokens (balance between granularity and context)
- **Overlap:** 50 tokens (maintains semantic continuity between chunks)
- - **Strategy:** RecursiveCharacterTextSplitter with hierarchical separators (paragraphs → sentences → words)

**Trade-off:** Larger overlap improves recall but increases storage. With unified pgvector, the overhead is manageable.

## Horizontal Scalability: What's Missing

The current architecture is intentionally a monolith. To scale horizontally, the following would be needed:

1. **Redis/Celery** for document processing queue (currently synchronous)
2. **PostgreSQL read replicas** for vector queries
3. **CDN** for frontend assets (currently served via Vite dev server)
4. **Load balancer** with sticky sessions for SSE

**Conscious decision:** Not implementing these components in the POC. Each adds operational complexity that is only justified after validating product-market fit.

## Use Cases Beyond Document Q&A

This architecture is not limited to document chat. It scales to:

- **Internal knowledge bases:** HR policies, engineering documentation, playbooks
- **Customer support automation:** With persisted conversation history
- **Audit trails for compliance:** Every query logged with its sources (source attribution)
- **Recommendation systems:** Similarity between documents and user profiles

## What I Learned About Trade-offs

**Cost vs. Complexity:** Managed solutions abstract complexity for a price. When you need total control (compliance, privacy), operational complexity becomes an investment, not a cost.

**Database Choice:** Unifying structured and vector data in PostgreSQL simplifies operations but limits some optimizations of specialized databases. For 99% of enterprise use cases, the performance difference is negligible compared to operational simplicity.

**Local Models vs. API:** Local models (Ollama) offer total privacy but require specialized hardware. External APIs (OpenRouter) democratize access but create dependency. The dual architecture allows for gradual migration as requirements evolve.

## Next Steps

The roadmap includes:

1. **Implementing MarkItDown** with fallback to traditional parsers in case of error
2. **LangGraph MVP** - migrate only the query pipeline initially
3. **Collect training data** for fine-tuning (real user Q&A)


---

**Applied AI Engineers:** What is your biggest concern when migrating from managed AI services to self-hosted pipelines — control, cost, or operational complexity?

---

*To see the full code and detailed technical documentation, access the project repository at [github.com/mabittar/local-rag](https://github.com/mabittar/local-rag)*
