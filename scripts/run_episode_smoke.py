#!/usr/bin/env -S uv run python
"""CLI smoke-test: run one full episode through ``SkillsGraphEnv``.

Drives a single rollout with uniformly random *legal* actions (filtered by
the action mask) so the env, action codec, and centrality scheduler all
exercise their public surface end-to-end. The script doubles as the
Phase-2 acceptance probe for two invariants from CLAUDE.md §CANONICAL
VALUES:

* ``Betweenness Centrality`` is called **exactly twice per seed** — once
  on ``reset`` (implicit, inside the env) and once when we ask for
  :meth:`SkillsGraphEnv.final_betweenness` (call 2 of 2). The assertion
  on ``env.centrality._betweenness_calls`` makes that contract loud.
* Action selection only ever samples from ``env.get_action_mask()``'s
  non-zero indices, which is the same masking rule the PPO head uses
  (Huang & Ontañón 2022; pre-softmax logit -> -inf).

Not part of the Phase-2 test suite — this is a hand-runnable probe.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.env.actions import global_index_to_action  # noqa: E402
from src.env.skills_graph_env import SkillsGraphEnv  # noqa: E402

DEFAULT_SEED = 42
DEFAULT_MAX_STEPS = 28
DEFAULT_SOURCE = REPO_ROOT / "src" / "pythonclaw_shim" / "sample_skills"
EXPECTED_BETWEENNESS_CALLS = 2  # ADR-006: exactly 2 per seed (start + end)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one SkillsGraphEnv episode with masked-random actions and "
            "verify the betweenness-centrality call-count invariant."
        ),
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Determinism seed.")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=DEFAULT_MAX_STEPS,
        help="Episode cap (truncation horizon).",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help="Skills source root for SkillsGraphEnv to graphify.",
    )
    return parser.parse_args(argv)


def _legal_indices(mask) -> list[int]:
    """Return list of legal flat action indices from a torch boolean/0-1 mask."""
    return mask.nonzero().squeeze(-1).tolist()


def _run_episode(env: SkillsGraphEnv, *, max_steps: int, seed: int) -> tuple[int, float]:
    """Drive the env for up to ``max_steps`` masked-random steps.

    Returns ``(steps_taken, total_reward)``. ``steps_taken`` is the number
    of completed ``env.step`` calls (1-indexed length, matching the spec's
    ``episode_length: t`` field).
    """
    rng = random.Random(seed)
    env.reset()  # state/info unused — env keeps its own canonical state
    total_reward = 0.0
    steps_taken = 0
    for t in range(max_steps):
        mask = env.get_action_mask()
        legal = _legal_indices(mask)
        if not legal:  # defensive — masking should always leave NOOP legal
            break
        chosen_idx = rng.choice(legal)
        action = global_index_to_action(chosen_idx)
        _state, reward, done, _info = env.step(action)
        total_reward += float(reward)
        steps_taken = t + 1
        if done:
            break
    return steps_taken, total_reward


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    env = SkillsGraphEnv(str(args.source), seed=args.seed)
    steps_taken, total_reward = _run_episode(env, max_steps=args.max_steps, seed=args.seed)

    # CALL 2 of 2 — see module docstring + ADR-006 + brief §2.2.
    final_betweenness = env.final_betweenness()
    betweenness_calls = env.centrality._betweenness_calls
    assert betweenness_calls == EXPECTED_BETWEENNESS_CALLS, (
        "Betweenness call-count invariant violated: "
        f"expected {EXPECTED_BETWEENNESS_CALLS}, got {betweenness_calls}"
    )

    final_v = env.current_state.num_nodes
    print(f"episode_length: {steps_taken}")
    print(f"total_reward: {total_reward:.6f}")
    print(f"betweenness_calls: {betweenness_calls}")
    print(f"final |V|: {final_v}")
    print(f"final_betweenness_nonzero: {sum(1 for v in final_betweenness.values() if v > 0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
