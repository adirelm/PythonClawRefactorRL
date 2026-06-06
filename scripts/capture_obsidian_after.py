"""CLI: render the post-PPO refactored Skills graph (brief §3 "after" shot).

Mirrors :mod:`scripts.capture_obsidian_stub` but, instead of loading the
baseline pickle, it replays the trained PPO policy for one episode at
``seed=42`` and dumps the *final* refactored graph as
``results/figures/obsidian_after.png``. Colour map, legend, and edge
styling are identical to the "before" shot so the two images are
diff-able by eye. Node-size formula is the Phase-1 cap ``min(LOC*8, 500)``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Headless backend; no display required.
import matplotlib.pyplot as plt
import networkx as nx
import torch
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.env.actions import global_index_to_action  # noqa: E402
from src.env.skills_graph_env import SkillsGraphEnv  # noqa: E402
from src.model.policy_net import PolicyNet  # noqa: E402
from src.services.ppo_trainer import _pad as _state_to_padded  # noqa: E402

_DEFAULT_CHECKPOINT = _REPO_ROOT / "results" / "training" / "seed_42" / "checkpoint.pt"
_DEFAULT_PNG = _REPO_ROOT / "results" / "figures" / "obsidian_after.png"
_DEFAULT_SOURCE = _REPO_ROOT / "src" / "pythonclaw_shim" / "sample_skills"
_SEED = 42
_LAYER_COLORS = {1: "lightblue", 2: "lightyellow", 3: "lightcoral", 0: "lightgrey"}
_LAYER_LABELS = {1: "L1 metadata", 2: "L2 instructions", 3: "L3 resources", 0: "code (module/class/fn)"}
_REL_COLORS = {"call": "#1f77b4", "import": "#ff7f0e", "inheritance": "#2ca02c"}
_FALLBACK_COLOR = "#999999"
_LOC_SCALE = 8
_LOC_CAP = 500
_PNG_MIN_BYTES = 1024  # anti-blank guard — a real labelled chart is well above this
_TITLE = "PythonClaw Skills shim — refactored dependency graph (AFTER PPO trained policy)"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Post-PPO refactored graph PNG (Obsidian stand-in).")
    parser.add_argument("--checkpoint", type=Path, default=_DEFAULT_CHECKPOINT, help="PPO checkpoint .pt")
    parser.add_argument("--output", type=Path, default=_DEFAULT_PNG, help="Output PNG path.")
    parser.add_argument("--source", type=Path, default=_DEFAULT_SOURCE, help="Skills source root.")
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


def _replay_episode(env: SkillsGraphEnv, policy: PolicyNet) -> None:
    """Run one greedy-policy episode in-place on ``env`` (uses action mask)."""
    state, _ = env.reset()
    done = False
    while not done:
        with torch.no_grad():
            x_padded, mask = _state_to_padded(state)
            logits, _ = policy(x_padded, mask)
            action_mask = env.get_action_mask().unsqueeze(0)
            action_idx, _ = policy.get_action(logits, action_mask)
        state, _reward, done, _info = env.step(global_index_to_action(int(action_idx.item())))


def _layer_key(raw: object) -> int:
    if isinstance(raw, int):
        return raw
    if isinstance(raw, str):
        token = raw.upper().lstrip("L")
        if token.isdigit():
            return int(token)
    return 0


def _styles(graph: nx.DiGraph) -> tuple[list[str], list[float], list[str]]:
    node_c = [
        _LAYER_COLORS.get(_layer_key(graph.nodes[n].get("layer")), _FALLBACK_COLOR) for n in graph.nodes
    ]
    node_s = [
        min(_LOC_SCALE * float(graph.nodes[n].get("LOC", 0) or 0), _LOC_CAP) or _LOC_SCALE
        for n in graph.nodes
    ]
    edge_c = [
        _REL_COLORS.get(str(d.get("rel_type", "")), _FALLBACK_COLOR) for _, _, d in graph.edges(data=True)
    ]
    return node_c, node_s, edge_c


def _legend_handles(graph: nx.DiGraph) -> list:
    layers = sorted({_layer_key(graph.nodes[n].get("layer")) for n in graph.nodes})
    rels = sorted({str(d.get("rel_type", "")) for _, _, d in graph.edges(data=True) if d.get("rel_type")})
    handles: list = []
    for lk in layers:
        handles.append(
            Patch(
                facecolor=_LAYER_COLORS.get(lk, _FALLBACK_COLOR),
                edgecolor="black",
                label=_LAYER_LABELS.get(lk, f"layer={lk}"),
            )
        )
    for rel in rels:
        handles.append(
            Line2D([0], [0], color=_REL_COLORS.get(rel, _FALLBACK_COLOR), lw=2, label=f"edge: {rel}")
        )
    return handles


def render(graph: nx.DiGraph, out_path: Path) -> None:
    """Draw refactored graph with spring_layout(seed=42) and write PNG."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pos = nx.spring_layout(graph, seed=_SEED)
    node_c, node_s, edge_c = _styles(graph)
    fig, ax = plt.subplots(figsize=(12, 9), dpi=140)
    nx.draw_networkx_nodes(
        graph, pos, node_color=node_c, node_size=node_s, ax=ax, edgecolors="black", linewidths=0.5
    )
    nx.draw_networkx_edges(graph, pos, edge_color=edge_c, arrows=True, ax=ax, alpha=0.7)
    nx.draw_networkx_labels(graph, pos, font_size=7, ax=ax)
    ax.legend(handles=_legend_handles(graph), loc="lower left", fontsize=8, framealpha=0.9)
    ax.set_title(_TITLE)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _verify_png(out_path: Path) -> None:
    """Anti-screenshot-debacle: confirm PNG exists and is non-trivial in size."""
    if not out_path.exists():
        raise RuntimeError(f"PNG not written: {out_path}")
    size = out_path.stat().st_size
    if size < _PNG_MIN_BYTES:
        raise RuntimeError(f"PNG suspiciously small ({size} B) — likely blank: {out_path}")
    with out_path.open("rb") as fh:
        header = fh.read(8)
    if header[:8] != b"\x89PNG\r\n\x1a\n":
        raise RuntimeError(f"Not a PNG header at {out_path}: {header!r}")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    env = SkillsGraphEnv(args.source, seed=_SEED)
    policy = _load_policy(args.checkpoint)
    _replay_episode(env, policy)
    render(env.graph, args.output)
    _verify_png(args.output)
    print(f"Figure written to {args.output} (size={args.output.stat().st_size} B)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
