---
layout: post
title: "The Scientific Harness for Self-Evolution: Measuring Agentic Systems Without Vibe Coding"
subtitle: "Transform hypotheses into verifiable facts through population-based loops, physical verification, and mathematical ROI"
date: 2026-09-01 08:00:00 +0300
description: "Scientific harness for agent self-evolution — abandon vibe coding and embrace rigorous measurement, execution traces, and mutation ROI."
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

## The Illusion of "Vibe Coding" in Agentic Systems

I was recently having a conversation with a colleague about the future of self-evolving agentic systems. And a brilliant question emerged:

> _How do you build an agent capable of evolving, equipped with scientific rigor, that can transform hypotheses into verifiable facts? That systematically analyzes, performs measurements, comparisons, and separates what's truly an improvement from mere noise?_

Today, we have sophisticated agentic systems. They don't just generate text — they _act_. They interact with terminals, repositories, APIs. They make decisions under uncertainty. And every decision is a hypothesis that needs testing and validation.

But here's the problem:

When a senior engineer says _"that prompt change improved quality"_, what did they actually measure?

- Personal feeling?
- A single benchmark (cherry-picked)?
- A comparison against "gold labels" (which always favor recent changes)?

**That's "vibe coding"** — and it's the silent enemy of agentic engineering.

---

## The Revelation: 1.6% AI + 98.4% Harness

Here comes the insight that changes everything.

Architectural analysis of modern agentic systems reveals a consistent pattern:

- **1.6% of code:** AI decision logic (LLM weights, semantics)
- **98.4% of code:** Operational infrastructure (orchestration, state management, fault recovery, physical verification)

This isn't a detail. It's _everything_.

Because it means:

```
A good harness with a weaker LLM
  >
A terrible harness with a stronger LLM
```

You can freeze the weights of a model and simply **mutate the harness surrounding it**, and observe empirically measured differences of up to **18 points in long-horizon benchmarks** and **6x in overall performance**. This is measured fact, not opinion.

**Model weights don't evolve during operation. The harness evolves.**

Therefore, the wrong question is: _"Which is the best model (GPT? Claude? Grok)?"_

The right question is: _"Which harness extracts maximum potential from the model we have?"_

---

## The Architecture: Three Roles, One Loop

To build a harness for self-evolution **without vibe coding**, we need an architecture based on **reproducible evidence**.

It's not a chaotic swarm. It's not a rigid graph (LangGraph). It's a **population-based loop guided by ROI**.

### **Role 1: SOLVER — The Physical Executor**

```
Responsibility: Execute the hypothesis (test) against the target system
               and collect DETERMINISTIC, REPRODUCIBLE EVIDENCE
```

The Solver doesn't interpret. Doesn't opine. Just executes and records. Transforms experiment into fact.

**Mandatory outputs:**

| Output                      | Description                                   | Why?                                          |
| --------------------------- | --------------------------------------------- | --------------------------------------------- |
| **Execution trace**         | Sequence of tool calls, timings, stack traces | Enables exact replication                     |
| **Verification Story**      | "Were files created? Contracts satisfied?"    | Tests if agent _actually_ did what it claimed |
| **Stderr/Stdout**           | Raw logs, parsed for error signals            | Separates real failures from false positives  |
| **Latency (P50, P95, P99)** | Distribution of time, not just average        | Identifies tail latency                       |
| **Token consumption**       | Prompt + completion, per step                 | Detects context leaks                         |
| **Success/Failure signal**  | Binary, based on physical verification        | Zero semantics, zero interpretation           |

**Illustrative pseudocode:**

```python
def solver_execute(harness_version: str, benchmark_case: str):
    """Execute application, collect reproducible physical evidence"""

    with isolated_environment(seed=42):  # Seed for reproducibility
        app = load_app_with_harness(harness_version)

        tracer = Tracer()
        start = perf_counter()

        try:
            result = app.run(benchmark_case)
            # PHYSICAL verification, not semantic
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

### **Role 2: PROPOSER — The Diagnostician Scientist**

```
Responsibility: Read Solver failures, propose DETERMINISTIC MUTATIONS
               in the harness (never in weights, never in prompt magic)
```

The Proposer is a "scientist" who examines the execution trace and formulates testable hypotheses.

**Typical mutation candidates:**

| Type                    | Change                                          | When to Apply                |
| ----------------------- | ----------------------------------------------- | ---------------------------- |
| **Context Strategy**    | Aggressive metadata compaction, lazy-load tools | Tokens used > 85% of limit   |
| **Retry Logic**         | Adaptive backoff vs. exponential vs. fail-fast  | Retry count > 3 in traces    |
| **Tool Ranking**        | Reorder tools by historical success frequency   | Tool success rate < 70%      |
| **State Checkpointing** | Persistence interval (every N steps?)           | Recovery latency critical    |
| **Error Recovery**      | Graceful degrade vs. exception propagation      | Specific failures recurring  |
| **Output Truncation**   | Limit tool outputs (1000 vs. 2000 tokens)       | Token overhead in tool calls |

**Real diagnostic example:**

You're analyzing execution traces from an agentic system. You observe:

- P95 latency up 40% in the past 7 days
- Retry count increased from 1.2 to 3.8 per session
- Tokens per call grew 22%

**Diagnosis:**

```
Hypothesis: Tool output is inflated, causing retry loops.
Proposed mutation:
  - Reduce tool_output_max_tokens from 2000 → 1000
  - Implement tool output summarization

Rationale: Trace shows 19% of latency comes from retry storms.
          If output is truncated, first attempt has more useful context.

Risk: Tool output might lose critical information.
Trade-off: Latency vs. information fidelity.
```

### **Role 3: JUDGE — The Arbiter (No Gold Labels)**

```
Responsibility: Select BEST mutation using PHYSICAL SIGNALS ONLY
               Forbidden: comparing against "correct answers"
```

This is the most critical role. Because here you **cannot cheat with semantic bias**.

The Judge sees only:

- ✅ Latency (improved?)
- ✅ Success rate (errors down?)
- ✅ Token efficiency (consumption down?)
- ✅ ROI (cost of change < gain?)

The Judge **never** sees:

- ❌ "The response seems prettier"
- ❌ "I have a good feeling about this"
- ❌ "Semantically, it's better"

**Judge pseudocode:**

```python
def judge_select_best_mutation(mutations: List[Mutation],
                               baseline: SolverOutput,
                               new_outputs: Dict[Mutation, SolverOutput]) -> Mutation:
    """Select winning mutation using ONLY physical signals"""

    scores = {}

    for mutation, output in new_outputs.items():
        # Criterion 1: Latency (P95)
        latency_gain = (baseline.latency_p95 - output.latency_p95) / baseline.latency_p95
        latency_score = max(0, latency_gain)  # 0 if worsened

        # Criterion 2: Success rate
        baseline_success = baseline.verification_story["contracts_met"]
        output_success = output.verification_story["contracts_met"]
        success_delta = output_success - baseline_success
        success_score = 1.0 if success_delta >= 0 else -0.5

        # Criterion 3: Token efficiency
        token_gain = (baseline.tokens_used - output.tokens_used) / baseline.tokens_used
        token_score = max(0, token_gain)

        # Criterion 4: ROI (gain vs. cost of mutation)
        mutation_cost = estimate_operational_cost(mutation)
        total_gain = (latency_score * 0.4 +
                      success_score * 0.3 +
                      token_score * 0.3)
        roi = total_gain - mutation_cost

        scores[mutation] = roi

    # Return mutation with highest ROI
    # NOT "the one that sounded best", but "the one that performed best"
    winner = max(scores.items(), key=lambda x: x[1])[0]
    return winner
```

---

## Criteria That Cannot Be Missed

### **1. Physical Tool Verification (T³RL)**

Not sufficient:

```
"The model generated a response"
```

Necessary:

```
"Tool was invoked → returned valid JSON →
 was parsed → affected filesystem state →
 transaction confirmed → integrity check passed"
```

**Implementation:**

```python
class PhysicalVerification:
    def verify_tool_execution(self, tool_call: ToolCall, result: Any) -> bool:
        """Validate that the tool truly executed with consequence"""

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

        # ALL must pass, not "most"
        return all(checks.values())
```

### **2. Execution Trace Analysis**

Not just the final answer. The path matters.

**Components of a useful trace:**

```
Tool call sequence
  ├─ What order? (reflects model priorities)
  └─ Correct callbacks? (state propagated?)

Token usage per step
  ├─ Where's the overhead?
  └─ Which tool consumes most?

Error recovery events
  ├─ How many failures?
  ├─ Which strategies worked?
  └─ Success rate by type?

State mutations
  ├─ Memory altered?
  ├─ Files created?
  └─ Transactions committed?

Latency breakdown
  ├─ LLM inference: X ms
  ├─ Tool execution: Y ms
  ├─ Parsing: Z ms
  └─ Retry loops: W ms (where to improve?)
```

**Real example:**

```
Failure trace in Claude Code v1.4:

Step 1: user_query → encoding           12ms  ✓
Step 2: tool_selection (LLM)            450ms ✓
Step 3: bash_execution                  5ms   ✗ [permission denied]
Step 4: error_recovery → retry+sudo     200ms ⚠️ [3 attempts]
Step 5: final_result (generation)       380ms ✓

Total latency: 1047ms
% from retries: 200/1047 = 19% (reducible!)

ACTION: Pre-validate permissions before execution
        Proposal: check `stat` before `rm`
```

### **3. Viability Equation (Total Cost)**

Not every mutation is worth evolving.

```
C_total = N_variants × N_traces × (C_prompt + C_generation) + C_judge

Where:
  N_variants = how many mutations to test
  N_traces = how many benchmarks per variant
  C_prompt = input tokens (Proposer analysis)
  C_generation = output tokens (proposals)
  C_judge = final selection tokens
```

**Numerical example (Hermes Agent analysis):**

```
N_variants: 5 mutations
N_traces: 20 test cases
C_prompt: 2000 tokens/trace
C_generation: 500 tokens/trace
C_judge: 1000 tokens
Price: $0.0075 per 1M tokens

C_total = 5 × 20 × (2000 + 500) / 1M × $0.0075
        + 1000 / 1M × $0.0075
        = 100 × 0.01875 + 0.0000075
        ≈ $0.38 per generation

Expected gain:
  - Latency reduced 15% → 1000 calls/day = 150s saved
  - Tokens reduced 12% → 50k tokens/day = 6k tokens saved
  - Value per token economized (in compute): ~$0.000001

  ROI ≈ (150s × micro_latency_value + 6k × $0.000001) / $0.38

Viable? Yes, if compound gain > 1000x cost.
```

### **4. Context as Bottleneck (Lazy-Loading)**

Context size is the scarcest resource in agentic systems.

**Observed compaction layers (real Claude Code):**

```
1. Metadata striping
   └─ Remove unnecessary IDs, non-critical timestamps

2. Tool schema on-demand
   └─ Load `tool.json` ONLY when tool is selected

3. Error alert aggregation
   └─ Summarize N identical errors into 1 log entry

4. Microcompression
   └─ Optimized tokenization for known data structures

5. LLM-based summarization
   └─ Last resort — subagent summarizes prior content
```

**Critical metric:**

```
Total available context:       100k tokens (Claude 3.5 Sonnet)
Context used per session:      89k tokens
Safety margin:                 11% (CRITICAL!)

Necessary action:
  ├─ Implement lazy-loading of tool schemas
  ├─ Threshold: load tool X ONLY if used in last 3 turns
  └─ Expected result: reduce to 75k tokens, margin = 25% (comfortable)
```

---

## Architecture: Population Loop vs. Swarm vs. Graph

Which is best? **None of them pure.**

### **Ideal Pattern: Population-Based Loop with Selective Breeding**

```
Generation N:
  Current population: [Harness_v1, Harness_v2, Harness_v3]

  ├─ Solver executes each version (3 × 20 traces) → 60 SolverOutputs
  ├─ Proposer analyzes common failures → 5-7 mutation candidates
  ├─ Judge selects top 3 mutations by ROI
  └─ Recombine: cross-breed winner + runner-up

  ↓

  Generation N+1: [Harness_winner, Harness_hybrid1, Harness_hybrid2]

Stopping criterion:
  - No mutation achieves ROI > 1.1x for 5 consecutive generations
  - OR cumulative C_total > evolution budget
  - OR algorithm converges (law of diminishing returns)
```

**Why not Swarm?**

- Swarms diverge chaotically, no clear direction
- Impossible to replicate result (too stochastic)
- Hard to prove a change "really helped"

**Why not rigid LangGraph?**

- Fixed paths prevent discovering novel patterns
- Each path is hardcoded; evolution limited to thresholds

**Why Population-Based Loop?**

- ✅ Deterministic (reproducible with seed)
- ✅ Converges on clear direction (mathematical ROI)
- ✅ Clear scientific proof: _"Mutation X won by 12% latency"_
- ✅ Allows recombination (genetic evolution with rigor)

---

## Model Weights vs. Harness: Mathematical Reality

You have an LLM with N billion parameters. How much does it actually matter?

```
Final Performance = f(Model_Weights, Harness)

Empirically (Claude Code + source code analysis):
  - Model weights: 99.8% fixed during operation
  - Harness: 98.4% of control surface (mutable)

∴ Impact of changes:
  - Alter weights (fine-tuning): ±2-5 points on benchmarks
  - Alter harness (context only): ±18 points (9x higher!)
  - Alter harness (retry + tool ranking): 6x overall performance
```

**What this means:**

```
LLM is the "thinker"     — decides which tool to use
Harness is the "nervous system" — decides:

  ├─ How many chances the LLM gets to succeed (retry logic)
  ├─ What context the LLM sees (compaction)
  ├─ What order tools appear (ranking)
  ├─ How errors are recovered (strategy)
  └─ When the LLM should stop (exit conditions)

Weak LLM with excellent harness > Strong LLM with terrible harness
```

---

## Applying It in Practice: Analyzing Hermes Agent

Suppose you want to evaluate **Hermes Agent** using this framework.

### **Phase 1: Baseline (Solver)**

Run Hermes against a benchmark of 20 cases:

- Solve coding problems
- Debug errors
- Optimize performance

Result: Hermes achieves 72% success, 3.2s average latency, 45k average tokens.

### **Phase 2: Diagnosis (Proposer)**

Examine failure traces from the 28% that failed:

```
Pattern 1: Poor tool selection
  └─ Hermes picks "grep" when should use "find"
     Success rate of "grep": 61%
     Success rate of "find": 89%

Pattern 2: Context overflow
  └─ After 5 iterations, context saturates
     Last step: 98k tokens used, truncation error

Pattern 3: Retry storms
  └─ When "bash execution" fails, Hermes retries 7x instead of 3x
     P95 latency: 8.5s (caused by retry loops)
```

Proposer generates 5 mutations:

1. **Tool ranking:** Reorder tools by historical success rate
2. **Context strategy:** Implement lazy-loading of tool schemas
3. **Retry logic:** Limit retries to 3, adaptive backoff
4. **Output truncation:** Reduce tool_output_max_tokens from 3000 → 1500
5. **State checkpointing:** Persist state every 4 steps

### **Phase 3: Selection (Judge)**

Run each mutation against the same benchmark:

| Mutation            | Success | Latency | Tokens | ROI            |
| ------------------- | ------- | ------- | ------ | -------------- |
| Baseline            | 72%     | 3.2s    | 45k    | —              |
| Tool ranking        | 78%     | 3.0s    | 44k    | **+8.3%** ✅   |
| Context strategy    | 72%     | 3.1s    | 38k    | +6.2%          |
| Retry logic         | 74%     | 2.5s    | 42k    | +12.1% ✅      |
| Output truncation   | 68%     | 3.0s    | 40k    | -2.3% (reject) |
| State checkpointing | 73%     | 3.4s    | 45k    | +1.5%          |

**Winner:** Retry logic (+12.1% ROI)

### **Phase 4: Next Generation**

Combine:

- Retry logic (winner) +
- Tool ranking (runner-up) +
- Partial context strategy (promising)

New population: [Hermes_winner, Hermes_hybrid1, Hermes_hybrid2]

Repeat cycle until convergence.

---

## Critical Trade-Offs

| Aspect              | High Precision      | High Speed     | Recommended Balance |
| ------------------- | ------------------- | -------------- | ------------------- |
| **N_traces**        | 100+ cases          | 10 cases       | 20-30 (sweet spot)  |
| **Retry strategy**  | Exponential backoff | Fail-fast      | Adaptive jitter     |
| **Tool output**     | 5000 tokens         | 500 tokens     | 1500 tokens         |
| **Checkpoint freq** | Every step          | Every 10 steps | Every 4 steps       |
| **Evolution iters** | 50+ generations     | 3 generations  | 10-15 generations   |

You **cannot optimize everything**. Pick 2 critical variables for your case.

---

## Next Steps

1. **Implement Solver:** Create function executing agent and collecting traces
2. **Implement Proposer:** Failure parser, deterministic mutation generator
3. **Implement Judge:** ROI-based selector (never vibes)
4. **Validate on Hermes Agent:** Run full cycle, document results
5. **Scale to Claude Code:** Apply framework to your own harness

---

**For agentic systems engineers:** Which variable would you optimize first — latency, success rate, or token efficiency? And how would you measure "real improvement" without falling into vibe coding?

---

_Next post: Practical implementation of T³RL (Test-Time Reinforcement Learning) with physical tool verification._
