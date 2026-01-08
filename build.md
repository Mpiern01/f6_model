supports Phi-4-style staged training, adds IQuest-style code-flow, InfTool closed-loop tool-use, Jan long-horizon execution stability, plus an optional safe model-fusion path. All “no-storage” and “Mac MLX 4-bit” constraints are first-class.

F5 StreamTrain: Phi-4-style, No-Storage Training Pipeline (Mac-first)

Internal Name: F5 StreamTrain
Primary Outcome: a single “F5 Developer + Ops Agent Model” (base: janhq/Jan-v2-VL-high) specialized for:

long-horizon execution in real software environments (browsers/desktop-like workflows) 
Hugging Face

secure tool use (MCP/function calling, codebase edits, test/build loops) 
arXiv

repo-evolution / “code-flow” learning from commits (IQuest-style) 
Hugging Face
+1

exportable to MLX and 4-bit for Mac deployment

1) What “for F5” means (product + security posture)
1.1 F5 alignment (what we’re building this for)

This pipeline produces a model we can ship and run:

inside customer-controlled environments (data sovereignty by default)

behind F5-managed ingress / policy (API gateway/WAF-style controls, identity, rate limiting, logging)

as an “agentic” assistant for:

secure code changes (repo + CI)

secure API/tool calling (internal APIs, IaC workflows)

incident/change workflows (ticket → plan → patch → verify → rollout)

We are not assuming any proprietary F5 datasets. This pipeline is built to train on public HF streamed data and customer-owned local data (optional), without persisting training datasets.

2) Non-negotiable constraint: “stream data, don’t store it”
2.1 Streaming guarantee

HF streaming explicitly allows iterating over datasets without downloading the entire dataset; data is streamed as you iterate. 
Hugging Face
+1

2.2 Cache reality & our policy

Even with streaming, some caching exists. HF documents:

datasets cache and how to change it (HF_HOME, HF_DATASETS_CACHE) 
Hugging Face

hub cache location and HF_HOME behavior 
Hugging Face

F5 StreamTrain policy

No dataset materialization: no local parquet/arrow builds; no “save_to_disk”.

Ephemeral caches only: route HF caches to an ephemeral volume (RAM disk or encrypted temp).

Only model artifacts persist: LoRA adapters, optimizer state, checkpoints, eval logs.

3) Base model target: Jan-v2-VL-high (Mac deploy lane)

Jan-v2-VL-high is designed for long-horizon, multi-step tasks in real software environments, combining language reasoning + visual perception and error recovery. 
Hugging Face

MLX conversion (officially reproducible)

The MLX community card provides a concrete conversion command for Jan-v2-VL-high: 
Hugging Face

pip install mlx-vlm
mlx_vlm.convert --hf-path janhq/Jan-v2-VL-high --dtype bfloat16 --mlx-path Jan-v2-VL-high-bf16-mlx

Why MLX-VLM matters for F5 StreamTrain

MLX-VLM supports inference and fine-tuning of VLMs and Omni models (audio/video) on Mac. 
GitHub

So we design a planner/executor harness where Jan is the planner and other modality executors plug in later (Section 10).

4) Training recipe: Phi-4-style staged pipeline (generalized to SWE)

We follow the 4-stage recipe from Phi-4-Mini-Reasoning:

large-scale mid-training on distilled long-CoT

SFT on high-quality long-CoT

Rollout DPO from curated prefs

RL with Verifiable Reward (RLVR) 
arXiv

This is our “spine,” but the data and verifiers are SWE/tooling oriented.

5) Data plane: what we stream (public) + what we synthesize (closed loop)
5.1 Code-flow (repo evolution): CommitPack commit format

We use bigcode/commitpack-subset-cf as the canonical “code-flow” stream; it’s explicitly in commit format:
<commit_before>code_before<commit_message>commit_message<commit_after>code_after 
Hugging Face
+1

5.2 IQuest-style “code-flow” behaviors we train for

IQuest’s model card highlights:

learning from repository evolution patterns, commit transitions, dynamic transformations

bifurcated post-training into Thinking vs Instruct variants 
Hugging Face

and their README emphasizes loop variants and native long context. 
Hugging Face

F5 StreamTrain implements the behaviors (not their proprietary pipeline):

commit-conditioned patching

“why this change?” commit-message reasoning

edit → test → fix loops

long-context repo reasoning

5.3 Closed-loop tool-use synthesis (InfTool pattern)

InfTool (Close the Loop) describes:

multi-agent role-play (User Simulator, Tool-Calling Assistant, MCP Server)

closed loop: synthesize → train with GRPO + gated rewards → synthesize better data → repeat 
arXiv
+1

This is our main “small data → big benchmark jump” lever.

5.4 Long-horizon SWE context management (“Context as a Tool”)

For long-horizon SWE agents, context loss is a primary failure mode. “Context as a Tool” targets explicit context management as a tool in the agent loop. 
arXiv

We incorporate a ContextTool API that the model must call (summarize, pin, retrieve, compress).

6) Model outputs: two official F5 variants (same base weights, different post-training)

We ship two checkpoints:

F5-StreamTrain-Instruct: concise, reliable, policy-compliant tool use

F5-StreamTrain-Thinking: deeper reasoning traces + longer-horizon planning

This mirrors IQuest’s “dual specialization paths” concept (Thinking vs Instruct). 
Hugging Face

7) The Stage Graph (what the team actually implements)
Stage 0 — Environment + ephemeral cache boot

Goal: guarantee “no dataset storage.”

Set HF cache env vars to an ephemeral mount:

HF_HOME

optionally HF_DATASETS_CACHE 
Hugging Face
+1

Stream datasets with streaming=True 
Hugging Face
+1

Stage 1 — Mid-training (continued pretraining) on code-flow + long-context

Objective: inject repo-evolution priors without catastrophic drift.

Online loss

𝐿
=
𝐿
𝐿
𝑀
+
𝜆
𝐾
𝐿
 
𝐾
𝐿
(
𝜋
𝜃
∥
𝜋
𝑏
𝑎
𝑠
𝑒
)
+
𝜆
𝑎
𝑛
𝑐
ℎ
𝑜
𝑟
 
𝐿
𝑎
𝑛
𝑐
ℎ
𝑜
𝑟
L=L
LM
	​

+λ
KL
	​

KL(π
θ
	​

∥π
base
	​

)+λ
anchor
	​

L
anchor
	​


KL keeps the model near Jan baseline (drift control)

anchor is an immutable set of regression prompts (SWE + tool schemas)

Data mix (streamed)

50–70% commitpack code-flow 
Hugging Face

15–25% long-context repo snapshots (packed)

10–25% synthetic tool traces (from Stage 5 loop)

Mac reality

LoRA-only training by default; gradient checkpointing; micro-batches.

Stage 2 — SFT (high quality SWE trajectories)

Format: (spec → plan → tool calls → patch → tests → final summary)
This stage builds deterministic behavior and tool schemas.

Stage 3 — Rollout-DPO (preference optimization from verifiers)

We generate K rollouts per prompt and label winners/losers by verifiers:

Verifiers

tests pass/fail

build/lint pass/fail

tool schema validity

patch applies cleanly

Then run DPO/IPO-style optimization (implementation choice).

Stage 4 — RLVR (GRPO-family)

Phi-style RL with verifiable reward is our capstone. 
arXiv

We strongly prefer a GRPO-family optimizer because it is widely used for RLVR in open reasoning model training; GRPO’s connection to DPO has also been analyzed in the literature. 
OpenReview
+1

Reward

𝑅
=
𝑤
1
1
[
tests pass
]
+
𝑤
2
1
[
build ok
]
+
𝑤
3
1
[
schema ok
]
−
𝑤
4
⋅
cost
R=w
1
	​

1[tests pass]+w
2
	​

1[build ok]+w
3
	​

1[schema ok]−w
4
	​

⋅cost

2025–2026 “push” upgrades for RLVR stability

MS-GRPO: reported improvements across reasoning/coding benchmarks in its OpenReview description 
OpenReview

INFO-GRPO: targets GRPO failure modes like collapse/singularities (OpenReview PDF) 
OpenReview

(We keep both as pluggable trainers in /train/rl/.)

Stage 5 — InfTool closed loop (infinite tool-use data)

Implement the three-agent loop described by InfTool, including MCP server simulation and gated rewards. 
arXiv
+1

This stage runs continuously and feeds new data back into Stages 2–4.

8) Test-time compute scaling + “loop” capability (frontier feel without bigger weights)

Two complementary mechanisms:

8.1 Test-time compute scaling for agents

“The Art of Scaling Test-Time Compute…” surveys strategies to improve reasoning by increasing inference-time compute. 
arXiv
+1

We implement an agent-side TTS controller:

sample N candidate plans

score via verifiers

pick best, iterate

8.2 Looped reasoning direction

Looped Language Models (LoopLM) show iterative computation as a scaling direction. 
arXiv
+1

For F5 StreamTrain we implement two levels:

Harness-level looping (always safe): multiple forward passes + tool feedback.

Architecture-level looping (only if compatible): optional recurrent-depth adapters.

9) Optional Model Fusion (only when models “line up”)

Goal: allow safe merges when architectures and tokenizers match (e.g., Jan-derived forks).

9.1 Compatibility gate (hard requirement)

Fusion is allowed only if:

same tokenizer vocab + special tokens

identical layer counts / hidden size / attention config

parameter name alignment (strict)

9.2 Supported fusion modes

LoRA fusion (safe default): weighted merge of adapters; rollback is trivial.

Weight-space merges (TIES/DARE-like): only in an offline “fusion lab” with regression gates.

Regression gates

anchor eval must not drop beyond threshold

tool schema validity must remain above threshold

long-horizon task success must not regress

10) “Full generative multimodal” harness (F5 deployment shape)

MLX-VLM explicitly targets Omni models (audio/video support) on Mac. 
GitHub

But Jan-v2-VL-high is primarily a VLM planner. 
Hugging Face

So we architect F5 Unified Model Surface:

10.1 Planner/Executor split

Planner: Jan-v2-VL-high (tool-using, vision-grounded)

Executors (pluggable):

image generator

audio generator

video generator

code runner / test runner

10.2 One-model UX promise

Externally, F5 exposes one endpoint (one “model”), but internally the harness routes to executors.

10.3 Security controls (F5-grade)

tool allowlists (per tenant)

schema enforcement (MCP server)

deterministic audit logs of tool calls + diffs

rate limits + anomaly detection

11) MLX + 4-bit deliverable (Mac deployment)
11.1 Conversion

Jan MLX conversion command is reproducible. 
Hugging Face

11.2 Quantization

HF’s MLX docs show mlx_lm.convert supports quantization flows for LLMs 
Hugging Face
 and Apple’s WWDC session describes converting + quantizing down to ~4 bits per weight using MLX tooling. 
Apple Developer

For VLMs we prefer staying inside MLX-VLM tooling when available, otherwise export weights and quantize with a controlled offline script.

12) Repo skeleton (F5 naming)
/f5-streamtrain
  /configs
  /streaming
    hf_stream.py
    mixers.py
    packer.py
    formats/
      commitpack_codeflow.py
      swe_trace.py
      tool_trace.py
  /stages
    s0_env_bootstrap.py
    s1_midtrain.py
    s2_sft.py
    s3_rollout_dpo.py
    s4_rlvr_grpo.py
    s5_inftool_loop.py
  /context_tool
    api.py
    compressor.py
    retriever.py
  /verifiers
    tests.py
    build.py
    schema.py
    patch_apply.py
  /fusion
    compat.py
    lora_fuse.py
    ties_dare_lab.py
  /mlx
    export_mlx_vlm.py
    quantize_4bit.py
  /eval
    long_horizon_suite.py
    swe_suite.py
  main.py
