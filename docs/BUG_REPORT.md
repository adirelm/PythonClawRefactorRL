# Bug Report — Real Bugs in PythonClaw

> **Brief §3 deliverable.** Genuine defects in the **real PythonClaw** source
> ([github.com/ericwang915/PythonClaw](https://github.com/ericwang915/PythonClaw),
> PyPI `pythonclaw`), pinned at commit `7787bb43` (v0.6.6). Reproduce the
> structural analysis with `scripts/fetch_pythonclaw.py` + `scripts/analyze_real_pythonclaw.py`.
>
> **Verification provenance.** GRAPHIFY reverse-engineering flagged the hub
> modules; a **30-agent hunt** (24 area×category hunters → 50 real candidates →
> 6 adversarial verifiers) then separated genuine defects from smells. The four
> security bugs (1–4) were each returned **CONFIRMED_REAL_BUG** (the God Object
> `(S)` is a *smell*, not counted as a bug); an earlier 20-agent pass confirmed
> Bug 1 (10/10) and labelled the God Object `SMELL_ONLY` (6/6). The three
> **Skills-module architectural failures (5–7)** were added from a focused
> `core/skill_loader.py` audit; findings 5 and 7 were **reproduced** (see each
> entry). All findings are **structural/security defects** (we do not execute
> PythonClaw). Every file:line is against the pinned source.

## Severity summary

**Brief §3 scope.** The brief asks for **≥2 architectural bugs/failures in the
Skills module** specifically. Bugs **3, 5, 6, 7** are all in the Skills subsystem
(`core/skillhub.py` + `core/skill_loader.py`) — four findings, of which **§7 is a
direct lazy-load-tiering violation** of the exact L1/L2/L3 design this project
models. The security bugs (1, 2, 4) live in the agent/tool/web layers and are
reported as additional genuine defects.

| # | Bug | Severity | Type | Skills module? |
|---|---|---|---|---|
| 1 | Command injection in `run_command` (`shell=True` on LLM input) | **CRITICAL** | CWE-78 RCE | — `core/tools.py` |
| 2 | Unauthenticated, network-exposed web dashboard → remote RCE | **CRITICAL** | CWE-306 / CWE-1327 | — `web/app.py` |
| 3 | Zip Slip path traversal in marketplace skill install | **CRITICAL** | CWE-22 → RCE | ✅ `core/skillhub.py` |
| 4 | Sandbox bypass: arbitrary file read (`read_file`) + exfiltration (`send_file`) | **HIGH** | CWE-22 / CWE-200 | — `core/tools.py` |
| 5 | No error boundary in discovery → one bad skill folder kills the whole catalog | **MEDIUM** | fault isolation | ✅ `core/skill_loader.py` |
| 6 | L1 metadata cache per-instance & discarded → disk re-scanned every call | **LOW** | lazy-load cache contract | ✅ `core/skill_loader.py` |
| 7 | L1 discovery eagerly reads the full L2 body — **lazy-load-tiering violation** | **MEDIUM** | broken tiering | ✅ `core/skill_loader.py` |
| (S) | God Object `core/agent.py` (1,151 LOC) | smell | anti-pattern, **not a bug** | — |

---

## Bug 1 (CRITICAL, CWE-78): command injection in the agent's `run_command` tool

`pythonclaw/core/tools.py` exposes `run_command(command)` as a **PRIMITIVE_TOOL
always advertised to the LLM** (registered `tools.py:236`; schema
`{"command":"The shell command to execute"}` `tools.py:270`). Its body runs the
model-supplied string through the shell:

```python
subprocess.run(command, shell=True, ..., timeout=60, env=_venv_env(), cwd=_files_dir())  # tools.py:151
```

`shell=True` on **LLM-controlled** input (or text injected via prompt injection — a
page the agent reads, a poisoned doc) → arbitrary command execution. The *same
file* sandboxes file ops (`_sanitize_filename`, `_sandbox_roots`,
`_resolve_in_sandbox`), but `run_command` has **no allow-list, no sandbox**. The
bundled `dev/code_runner` skill does it correctly (arg-list, no shell). **Fix:**
drop `shell=True` (`shlex.split` + arg vector) or gate behind an allow-list /
confirmation. *(20/20 + 30-agent verifiers: CONFIRMED.)*

---

## Bug 2 (CRITICAL, CWE-306 + CWE-1327): unauthenticated, network-exposed web dashboard → remote RCE

`create_app()` (`web/app.py:55-103`) registers **every** route with **no
authentication** — `GET/POST /api/config` (76-77), `/api/channels/restart` (98),
`/api/files/clear` (99), `/api/files` (100), and the **`/ws/chat` WebSocket**
(101). There is no `Depends`, `HTTPBearer`, API-key check, session check, or
auth middleware anywhere under `web/`.

Three compounding defects make this a full **remote RCE**:

- **No auth + bound to all interfaces.** `main.py:165` defaults `web.host` to
  `"0.0.0.0"` (`uvicorn.run`, `main.py:183`); `onboard.py:362` seeds the same. So
  the dashboard is reachable by **any host on the network**, not just localhost.
- **The chat endpoint reaches the shell.** `_ws_chat` (`web/app.py:803-905`)
  `await websocket.accept()`s any connection and forwards text to
  `agent.chat_stream` (`:888`) → the agent's `run_command` (Bug 1) → shell.
  **Unauthenticated remote shell via natural-language chat.**
- **CSWSH (Cross-Site WebSocket Hijacking, CWE-1385).** `_ws_chat` accepts the
  handshake with **no `Origin` check** (`web/app.py:804`). Browsers do not apply
  same-origin policy to WS handshakes, so *any malicious page a victim visits* can
  `new WebSocket("ws://localhost:7788/ws/chat")`, drive the agent, and read the
  streamed responses back — RCE + exfiltration even when bound to loopback.
- **Config/secret overwrite.** `POST /api/config` → `_api_config_save`
  (`web/app.py:206-261`) writes attacker JSON to `pythonclaw.json` (`:238`) and
  hot-reloads the LLM provider (`:248`) — also unauthenticated.

**Fix:** default `web.host` to `127.0.0.1`; require a per-install token on all
`/api/*` routes **and** the `/ws/chat` handshake; validate `Origin` before
`websocket.accept()`. *(3 independent verifiers: CONFIRMED CRITICAL.)*

---

## Bug 3 (CRITICAL, CWE-22 → RCE): Zip Slip in marketplace skill install

`core/skillhub.py` downloads a skill ZIP from the ClawHub marketplace and
extracts it without validating member paths:

```python
for member in zf.namelist():                 # skillhub.py:235 — from the downloaded ZIP
    if member.startswith("__MACOSX") or member.startswith("."):
        continue
    dest = os.path.join(skill_dir, member)   # skillhub.py:238 — NO check dest stays under skill_dir
    ...
        f.write(zf.read(member))             # skillhub.py:244 — write-anywhere
```

A malicious/compromised marketplace skill with a member like
`../../../../.config/systemd/user/evil.service` (or a cron file, shell rc, etc.)
**escapes `skill_dir` and writes anywhere the user can write** → persistence /
RCE. The only filter is a `__MACOSX`/dot-prefix skip — not path traversal. There
is no `os.path.realpath(dest).startswith(realpath(skill_dir))` guard. The same
sink is **duplicated** in `install_skill_async` (`skillhub.py:381-390`), which is
reachable from the **unauthenticated** `/api/marketplace/install` web route
(`web/app.py:587`, mounted `:91`) — and the ZIP download disables TLS verification
(`httpx … verify=False`, `skillhub.py:277,302`), so even a "trusted" CDN response is
MITM-substitutable. **Fix:** reject any member whose resolved destination is outside
`skill_dir` before writing (both sync + async paths); restore TLS verification.
*(Verifier: CONFIRMED CRITICAL.)*

---

## Bug 4 (HIGH, CWE-22 + CWE-200): sandbox bypass — arbitrary file read & exfiltration

Two LLM-facing primitives skip the sandbox the *write* path enforces:

- **`read_file(path)`** (`tools.py:160-166`) just `open(path)`s **any path** — no
  `_resolve_in_sandbox`. `read_file("~/.ssh/id_rsa")` / `/etc/passwd` returns the
  contents to the model.
- **`send_file(path, caption)`** (`tools.py:210-220`) resolves the path and checks
  existence + size, but **never checks it is inside the sandbox roots** (unlike
  `write_file`, which calls `_resolve_in_sandbox`, `tools.py:177`). So
  `send_file("/any/secret")` **exfiltrates any file to the external channel**
  (Telegram/Discord/WhatsApp/Web).

Together (and reachable via Bug 1/Bug 2 prompt-injection) these form a full
read-then-exfiltrate primitive that bypasses the sandbox-root check `write_file`
enforces. **Scope (honest):** the configured sandbox is `set_sandbox([PYTHONCLAW_HOME,
~])` (`core/agent.py:184`) — the *entire home dir* is already in-bounds by design,
so the crown-jewel files (`~/.ssh`, dotfiles, `~/.pythonclaw` secrets) are reachable
even *with* the sandbox. The missing `read_file`/`send_file` check therefore adds
arbitrary read/exfil of paths **outside `$HOME`** (`/etc/passwd`, other users' files)
on top of that, and removes the write/read asymmetry — a genuine defect, scoped
accurately rather than as "reads everything." **Fix:** route `read_file` and
`send_file` through `_resolve_in_sandbox` like `write_file` (and tighten the sandbox
roots below `~`). *(2 verifiers: CONFIRMED — real defect, impact scoped to extra-`$HOME` reads.)*

---

# Skills-module architectural failures (brief §3)

These three are **architectural failures in the Skills module itself**
(`core/skill_loader.py` — the L1/L2/L3 progressive-disclosure loader), distinct
from the Zip-Slip security bug (Bug 3, also in the Skills module). They are the
on-theme deliverable for brief §3: the project's whole premise is the tiered
Skills lazy-load architecture, and finding **5** and **7** are reproduced defects
*in that very mechanism*.

## Bug 5 (MEDIUM, fault isolation): no error boundary in discovery — one bad skill folder kills the whole catalog

`SkillRegistry.discover()` walks every skills directory with **no per-skill
guard**, and the inner category scan does a raw, unwrapped `os.listdir`:

```python
for s_dir in self.skills_dirs:            # skill_loader.py:155 — no try/except
    if not os.path.isdir(s_dir): continue
    self._scan_dir(s_dir, skills, seen_names)
...
for sub_entry in sorted(os.listdir(entry_path)):   # skill_loader.py:193 — unguarded
```

The only error handling in the L1 path is `_read_metadata`, which catches **only
`OSError`** (`skill_loader.py:273`). Anything escaping `_scan_dir` aborts the whole
scan. **Reproduced:** with three skill folders where the middle category dir is
unreadable (`chmod 000`), `discover()` raised `PermissionError` and returned
**zero** skills — the two valid skills were never discovered. Because
`Agent._init_system_prompt` calls `self._registry.build_catalog()`
(`core/agent.py:295`, `:406`) with no surrounding `try`, this propagates into
**agent boot**. A marketplace loader importing arbitrary third-party skills must
contain a single bad skill; this one does not. **Fix:** wrap each `_scan_dir`
call and per-entry body in `try/except Exception` → log-and-skip; degrade to "the
skills that parsed," never all-or-nothing.

## Bug 6 (LOW, broken lazy-load cache contract): L1 cache is per-instance and silently discarded

The metadata cache lives only on the instance (`skill_loader.py:125`;
`discover()` advertises "results are cached after the first call",
`skill_loader.py:146`), but the three public module-level entry points each build
a **throwaway** registry, so the cache never survives:

```python
return SkillRegistry(skills_dirs).load_skill(skill_name)   # skill_loader.py:376
for s in SkillRegistry(skills_dirs).discover()             # :387 and :403
```

`web/app.py:274` likewise constructs a fresh `SkillRegistry` per request, and
there is no shared singleton / `lru_cache`. So every `search_skills` /
`load_skill_by_name` / web `/skills` call re-walks the tree **and re-`open()`s
every `SKILL.md`** (`_read_metadata`, `:246-247`) from scratch — `O(calls × skills)`
instead of `O(skills)`. The cache only ever helps the single long-lived
`Agent._registry` (`core/agent.py:294`). **Fix:** expose one shared/memoized
`get_registry(skills_dirs)` that the module helpers and web layer reuse, with an
explicit `invalidate()` after install.

## Bug 7 (MEDIUM, lazy-load-tiering violation): L1 discovery eagerly reads the full L2 body off disk

`discover()` documents itself as the cheap metadata pass — *"only YAML frontmatter
is read (name + description)"* (`skill_loader.py:147`). But `_read_metadata` slurps
the **entire file** before parsing:

```python
with open(md_path, "r", encoding="utf-8") as f:
    content = f.read()               # skill_loader.py:246 — whole file
meta, _ = parse_frontmatter(content) # :248 — body parsed, then discarded ("_")
```

`parse_frontmatter` returns `(metadata, body)`; the **body is the Level-2
instruction text** that `load_skill` (`:279`) / the agent's `use_skill`
(`core/agent.py:588`) are supposed to read *on demand*. Materializing it during
L1 performs the L2 tier's disk I/O for **every** installed skill at startup — the
exact eager-loading the 3-tier design exists to prevent. It is the most direct
contradiction of the L1/L2/L3 contract this project models. (It does not leak into
the LLM context — only the disk read is eager — hence MEDIUM.) **Fix:** in
`_read_metadata`, read only up to the closing `---` of the frontmatter (bounded
prefix / line-by-line until the second delimiter) so the L2 body is never touched
during L1.

---

## (S) Not a bug — God Object `core/agent.py` (smell)

`core/agent.py` is 1,151 LOC and wires 27 collaborators in one `Agent.__init__`
— a real **anti-pattern** (maintainability), surfaced as the top fan-out node by
GRAPHIFY. It is **not a functional defect**; both verification passes labelled it
`SMELL_ONLY` (6/6). Listed for completeness, honestly as a smell. Coupling to
`core/llm/base.py` (fan-in 13, of which 3 are healthy client-implements-base) and
22/72 oversized modules are likewise smells, not bugs. PythonClaw has **no import
cycles and no dead code**.

---

## Appendix — engineering defects we found & fixed in our own RL pipeline
Not PythonClaw bugs. (A1) `Categorical(all_-inf)` NaN on empty action mask — fixed
`5dd14ca`. (A2) Louvain wedge on degenerate topologies — fixed RC-4 (SIGALRM cut +
stored masks). Regression tests `test_policy_net_categorical_safe.py`,
`test_modularity_wedge_regression.py`.

## Cross-references
- Reproduction: `scripts/fetch_pythonclaw.py`, `scripts/analyze_real_pythonclaw.py`
- Structural evidence: `results/data/real_pythonclaw_analysis.json`, `results/graphify_output.gpickle`
- Verification: 30-agent hunt (`wf_c184c556`) + 20-agent verify (`wf_211d7217`)
- Source decision: `docs/adr/ADR-001-pythonclaw-shim-boundary.md`
