---
title: "From Chatbot to Agent: Building a Conversational PIX Assistant with LangGraph and FastAPI"
date: 2026-07-16
draft: false
tags: ["langgraph", "fastapi", "llm", "ai-engineering", "python", "openrouter", "ollama", "observability"]
img: langgraph.png
categories: ["AI Engineering"]
---

## The Gap Between Chatbots and Agents

Most LLM demos stop where things start to get interesting: the model generates a nice text response, and that's it. But the real value of an LLM inside a product is not what it _says_ — it's what it _does_ after speaking.

Over the last few months I've been building a side project that deliberately crosses that line: a **conversational PIX assistant** that interprets natural language commands in Portuguese and executes real banking operations. The user types _"list my pix keys"_ or _"pay this QR Code"_, and the system classifies the intent, routes it through a state graph, calls the banking API, and answers with a humanized message — with authentication, caching, persistence, guardrails, and full observability along the way.

This post walks through the architecture and the trade-offs behind each decision. All code snippets are real, taken straight from the repository.

## Context: From the Classroom to a Side Project

The timing wasn't accidental. I'm currently enrolled in a post-graduation program in **Applied AI Engineering**, and the module on LLM API integration (taught by [Erick Wendel](https://github.com/ErickWendel), who actively encourages students to build in public) introduced exactly the tools I had been wanting to try hands-on: **LangGraph** for agent orchestration and **OpenRouter** for multi-model access.

At the same time, my day job is building Banking-as-a-Service products. Combining the two worlds was the natural move: take the agent patterns from the classroom and stress-test them against the kind of integration complexity I see every day in banking — JWT authentication, idempotency, transactional state, audit trails.

The result is two repositories:

- **[banking-llm](https://github.com/mabittar/banking-llm)** — the backend: a FastAPI service hosting a LangGraph state machine
- **[chat-llm](https://github.com/mabittar/chat-llm)** — the frontend: a Vue 3 chat SPA that talks to it

## What the System Does

A single endpoint, `POST /chat`, receives a message in natural language and returns a humanized answer. Behind it, six intents are supported:

| Intent           | Description                                |
| ---------------- | ------------------------------------------ |
| `list_keys`      | Lists active PIX keys on the account       |
| `read_key`       | Fetches details of a specific PIX key      |
| `pix_withdraw`   | Executes a PIX transfer to a key           |
| `brcode_preview` | Decodes and validates a PIX QR Code        |
| `pix_payment`    | Full QR Code payment (preview + transfer)  |
| `guardrail`      | Safety validation against prompt injection |

## Architecture Overview

The service follows a layered design, with the LangGraph machine sitting between the HTTP layer and the domain services:

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

Now let's get into the decisions that actually mattered.

## Decision 1: A State Machine, Not a Free-Form Agent

The first architectural fork in the road: should this be a ReAct-style agent that autonomously picks tools, or an explicit state graph?

| Approach                    | Pros                                                                                | Cons                                                     | Decision  |
| --------------------------- | ----------------------------------------------------------------------------------- | -------------------------------------------------------- | --------- |
| **LangGraph state machine** | Deterministic routing, auditable path, per-node observability, easy to replay/debug | More upfront design, less "emergent" behavior            | ✅ Chosen |
| ReAct agent (tool calling)  | Flexible, less code, handles unplanned paths                                        | Non-deterministic, hard to audit, hard to guarantee flow | ❌        |
| Pure prompt chaining        | Simple                                                                              | No conditional logic, brittle                            | ❌        |

**The key insight:** when the "tool" on the other side is a real money movement API, _non-determinism is not a feature — it's a liability_. I need to guarantee, by construction, that every request passes the guardrail before anything touches the banking API, and that a payment only happens after a preview. A state graph gives me those guarantees as edges, not as hopes baked into a system prompt.

The flow is explicit:

```
START ──▶ guardrail ──▶ identifyIntent ──(conditional)──▶ listKeys ──────────▶ chatResponse ──▶ END
                                              │                                      ▲
                                              ├──▶ readKey ──────────────────────────┤
                                              ├──▶ pixWithdraw ──────────────────────┤
                                              ├──▶ brcodePreview ────────────────────┤
                                              ├──▶ pixPayment ───────────────────────┤
                                              └──▶ fallback ─────────────────────────┘
```

## Decision 2: The Guardrail Is the First Node, Not a Middleware

Every request enters through the `guardrail` node, powered by a dedicated safety model (`llama-guard3:8b`). If the input looks like prompt injection, the graph short-circuits to a `blockedResponse` node and terminates — the intent classifier and the banking API never even see the message:

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

**Trade-off:** an extra LLM call on every message adds latency (visible in the `graph_node_duration_seconds` metric). But in a system where a successful injection could trigger a financial transaction, a dedicated safety gate at the _graph_ level — independent from the routing LLM — is non-negotiable. Defense in depth beats prompt engineering.

## Decision 3: LLM-as-Router with Structured Output

Intent classification is done by the LLM itself, but constrained through structured output. The state defines the contract as a `Literal` type:

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

Routing is then plain, testable Python — no LLM involved in the decision itself:

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

This split — **LLM proposes, deterministic code disposes** — is the pattern I'd recommend for any production agent: let the model do what it's good at (understanding messy natural language), and let code do what it's good at (enforcing control flow).

Notice the `brcode_ambiguous` command: when the LLM detects a BRCode but can't tell what the user wants to do with it, it routes to a safe fallback instead of guessing. Ambiguity is a first-class outcome, not an error.

## Decision 4: Dual LLM Provider — Ollama for Dev, OpenRouter for Prod

The `LLMService` abstracts the provider, so the same codebase runs in two modes:

| Mode        | Provider           | Model                         | Why                                                          |
| ----------- | ------------------ | ----------------------------- | ------------------------------------------------------------ |
| Development | **Ollama** (local) | `Qwen2.5:14B-Instruct-Q4_K_M` | Free, offline, private — iterate without burning API credits |
| Production  | **OpenRouter**     | `google/gemini-2.5-flash`     | Managed, fast, one API key for many models                   |

The guardrail model (`llama-guard3:8b`) also runs locally via Ollama. Swapping providers is a configuration change, not a refactor — the same dependency-injection principle I applied in my [local RAG project](https://mabittar.github.io/Local-RAG-Architecture-Eng/).

## Decision 5: Stateful Conversations Enable Real Payment Loops

Here's where LangGraph's checkpointer shines. PIX QR Codes can be issued _without an amount_ — the payer decides how much to pay. Handling that requires a multi-turn conversation with memory:

1. **User:** _"pague esse pix copia e cola 00020126...Jesse Pinkman..."_
2. **Assistant:** _"The QR Code you're trying to pay belongs to Jesse Pinkman. It has no specific amount associated. How much do you want to pay?"_
3. **User:** _"pague R$ 5.00"_
4. **Assistant:** _"Payment completed successfully. End-to-End ID: E49288113... — Amount paid: R$ 5.00 — Status: CREATED"_

State is persisted in PostgreSQL via `AsyncPostgresSaver`, keyed by a `thread_id`. The HTTP contract is simple: the client sends an `X-Thread-ID` header; if absent, the API generates one and returns it in the response header for the next turn.

The interesting part is how the intent node handles continuation — it checks the persisted state _before_ calling the LLM:

```python
def _handle_continuation(state: GraphState, messages: list) -> dict | None:
    """Resolve a pending amount from the user's last message, if any."""
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

**The key insight:** if the user is answering a pending question ("R$ 5.00"), you don't need an LLM call to figure out the intent — the state already tells you. A regex and a flag beat a token-spending classification, and the flow becomes deterministic exactly where money is involved.

## Decision 6: FastAPI as the Backbone

FastAPI was a deliberate choice, and it carries more weight here than just "serving HTTP":

- **Async end-to-end:** the graph invocation (`graph.ainvoke`), the banking client, Redis, and Postgres are all async — a sync framework would be a bottleneck pretending to be a server.
- **Pydantic v2 everywhere:** request/response contracts (`ChatRequest`, `ChatResponse`), settings via `pydantic-settings`, and DTOs at the banking boundary. One validation model from edge to edge.
- **Dependency injection via `app.state`:** cache and checkpointer are attached at lifespan startup and pulled inside the route — no globals.
- **Free OpenAPI docs:** the interactive Swagger UI at `/docs` doubles as a test harness for the whole graph.

The route itself stays thin — all complexity lives behind the graph:

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

## Decision 7: Observability Is Not Optional

An agent that calls external APIs and moves money without traces is a debugging nightmare waiting to happen. The service is instrumented with **OpenTelemetry** (opt-in via `OTEL_ENABLED`, no-op when off):

- **Traces:** root HTTP span, domain spans per graph node (`graph.intent.identify`, `pix.withdraw.execute`, ...), IO spans (banking API, Redis, Postgres), LLM spans with token counts. Exported to **Tempo/Jaeger**.
- **Metrics:** RED for HTTP plus domain metrics — `pix_operations_total`, `graph_node_duration_seconds`, `guardrail_block_total` — pushed to **Prometheus**.
- **Logs:** `structlog` JSON with automatic `trace_id`/`span_id` correlation, shipped to **Loki**.
- **Privacy:** PIX keys, government IDs, tokens, and secrets are masked in span attributes, metric labels, and logs.

The Grafana stack comes with pre-provisioned dashboards ("Service Overview", "PIX Operations"), so I can watch per-node p95 latency, guardrail block rates, and LLM token throughput while chatting with the assistant.

My favorite detail: the stack includes a **read-only Grafana MCP server** — which means another LLM agent can query the telemetry of _this_ agent. Agents observing agents, with a Viewer-only service account.

## The Frontend: A Deliberately Thin SPA

The chat UI ([chat-llm](https://github.com/mabittar/chat-llm)) is a Vue 3 + Vite SPA with Pinia and TailwindCSS. Its only real responsibility beyond rendering is the thread contract: generate a local UUID for a new conversation, send it as `X-Thread-ID`, and adopt whatever thread ID the API returns. Unit tests with Vitest, E2E with Playwright. Keeping the frontend thin keeps the conversational logic exactly where it belongs: in the graph.

## What I Learned About Trade-offs

**Determinism vs. autonomy.** The industry hype is on fully autonomous agents. For regulated, money-touching domains, the sweet spot is the opposite: LLMs for understanding, state machines for deciding. Autonomy is a spectrum, and you choose where to sit on it per use case.

**LLM calls are a budget.** The continuation handler saves one classification call per multi-turn payment. Small win? Sure. But designing flows where the _state_ answers before the model is a habit that compounds — in latency, cost, and predictability.

**Guardrails belong in the architecture, not in the prompt.** A system prompt that says "don't follow malicious instructions" is a suggestion. A graph edge that physically prevents reaching the banking API is a control.

**Local models are production-grade for dev loops.** Qwen2.5:14B quantized handles Portuguese intent classification and response generation surprisingly well. The entire dev cycle runs offline at zero marginal cost — OpenRouter only enters when I need managed scale.

## What's Next

The roadmap is public in the repo: authentication on `/chat` (after which the assistant can proactively fetch the customer's accounts), more PIX operations, and a CI/CD pipeline. The foundations — guardrails, persistence, observability — are already in place.

---

**Applied AI Engineers:** when your agent's tools can trigger real-world side effects, where do you draw the line between model autonomy and deterministic control flow?

---

_Full code, ADRs, and setup instructions: [github.com/mabittar/banking-llm](https://github.com/mabittar/banking-llm) · Frontend: [github.com/mabittar/chat-llm](https://github.com/mabittar/chat-llm)_
