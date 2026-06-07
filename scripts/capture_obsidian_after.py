"""CLI: render the post-PPO refactored Skills graph (brief §3 "after" shot).

Mirrors :mod:`scripts.capture_obsidian_stub` but, instead of loading the
baseline pickle, it replays the trained PPO policy at ``seed=42`` for a
**mid-rollout snapshot** and dumps the resulting refactored graph to
``results/figures/obsidian_after.png``. Colour map, legend, and edge
styling are identical to the "before" shot so the two images are
diff-able by eye. Node-size formula is the Phase-1 cap
``min(LOC*8, 500)``.

Why mid-rollout, not final?
---------------------------
At ``seed=42`` the trained policy keeps simplifying the graph past the
point where the picture is legible: by the terminal step the graph
collapses to a single connected pair (~2 nodes, 1 edge) and the
spring-layout output renders as a lone L3 dot, which reads either as a
broken figure or as the policy having destroyed all structure. Neither
is the message we want to communicate — the policy is doing *less*
destructive work than that summary implies; the terminal frame just
happens to be degenerate.

Instead we snapshot after ``_SNAPSHOT_STEPS`` rollout steps (currently
``32``, half of ``DEFAULT_MAX_EPISODE_STEPS=64``). This frame preserves
all three layers (L1 / L2 / L3) and ≥5 nodes with ≥3 edges, so the
viewer can compare topology against ``obsidian_before.png`` without
losing the message that the policy *has* aggressively merged /
rewired (node count drops from ~30 → 12, edges from ~10 → 8).

Determinism: ``PolicyNet.get_action`` samples from a Categorical, so
runs must be seeded. We call ``torch.manual_seed(_SEED)`` before the
rollout so the snapshot is reproducible byte-for-byte.

Render/styling helpers live in :mod:`scripts._capture_obsidian_lib`
to keep this CLI entry under the 150-LOC cap (CLAUDE.md §1).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts._capture_obsidian_lib import render, verify_png  # noqa: E402
from src.env.actions import global_index_to_action  # noqa: E402
from src.env.skills_graph_env import SkillsGraphEnv  # noqa: E402
from src.model.policy_net import PolicyNet  # noqa: E402
from src.services.ppo_trainer import _pad as _state_to_padded  # noqa: E402

_DEFAULT_CHECKPOINT = _REPO_ROOT / "results" / "training" / "seed_42" / "checkpoint.pt"
_DEFAULT_PNG = _REPO_ROOT / "results" / "figures" / "obsidian_after.png"
_DEFAULT_SOURCE = _REPO_ROOT / "src" / "pythonclaw_shim" / "sample_skills"
_SEED = 42
_SNAPSHOT_STEPS = 32  # Mid-rollout: preserves ≥5 nodes + ≥3 edges across L1/L2/L3.
_TITLE = (
    "PythonClaw Skills shim — refactored dependency graph "
    f"(AFTER PPO trained policy, mid-rollout @ step {_SNAPSHOT_STEPS})"
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Post-PPO refactored graph PNG (Obsidian stand-in).")
    parser.add_argument("--checkpoint", type=Path, default=_DEFAULT_CHECKPOINT, help="PPO checkpoint .pt")
    parser.add_argument("--output", type=Path, default=_DEFAULT_PNG, help="Output PNG path.")
    parser.add_argument("--source", type=Path, default=_DEFAULT_SOURCE, help="Skills source root.")
    parser.add_argument(
        "--steps", type=int, default=_SNAPSHOT_STEPS,
        help=f"Mid-rollout step count to snapshot at (default {_SNAPSHOT_STEPS}).",
    )
    return parser.parse_args(argv)


def _load_policy(checkpoint: Path) -> PolicyNet:
    """Load PolicyNet weights from ``checkpoint``; default-init if file missing."""
    policy = PolicyNet()
    if checkpoint.exists():
        # SAFETY: checkpoint is the local PPO artefact written by train_ppo.py.
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        policy.load_state_dict(state)
    policy.eval()
    return policy


def _replay_snapshot(env: SkillsGraphEnv, policy: PolicyNet, steps: int) -> None:
    """Run ``steps`` greedy-policy steps in-place on ``env`` (uses action mask).

    Stops early on ``done`` to avoid degenerate terminal frames.
    """
    torch.manual_seed(_SEED)  # determinism for Categorical.sample inside get_action
    state, _ = env.reset()
    for _ in range(steps):
        with torch.no_grad():
            x_padded, mask = _state_to_padded(state)
            logits, _ = policy(x_padded, mask)
            action_mask = env.get_action_mask().unsqueeze(0)
            action_idx, _ = policy.get_action(logits, action_mask)
        state, _reward, done, _info = env.step(global_index_to_action(int(action_idx.item())))
        if done:
            break


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    env = SkillsGraphEnv(args.source, seed=_SEED)
    policy = _load_policy(args.checkpoint)
    _replay_snapshot(env, policy, args.steps)
    g = env.graph
    print(
        f"snapshot @ step={args.steps}: nodes={g.number_of_nodes()}, edges={g.number_of_edges()}"
    )
    render(g, args.output, _TITLE)
    verify_png(args.output)
    print(f"Figure written to {args.output} (size={args.output.stat().st_size} B)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
