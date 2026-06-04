"""Architectural contract: NO references to stale split-config filenames.

CLAUDE.md §4 single-source-of-truth: the project uses ONE config file —
`config/config.yaml` — with sub-blocks accessed via `config/config.yaml#<block>`
notation. References to the old split-config filenames are stale and MUST be
swept from docs/.

Stale filenames (forbidden):
    config/state.yaml
    config/action.yaml
    config/reward.yaml

This test is ACTIVE (no xfail) — it is the gate other agents' doc-sweeps
must clear before Phase 1 closes.
"""

from __future__ import annotations

from pathlib import Path

STALE_REFS = (
    "config/state.yaml",
    "config/action.yaml",
    "config/reward.yaml",
)


def _scan_for_stale_refs(root: Path) -> list[str]:
    """Return a list of "<path>:<lineno>: <line>" hits for any stale ref."""
    hits: list[str] = []
    if not root.exists():
        return hits
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        # Only inspect text-like files we author (md, rst, txt, py, yaml).
        if path.suffix.lower() not in {".md", ".rst", ".txt", ".py", ".yaml", ".yml"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            for stale in STALE_REFS:
                if stale in line:
                    hits.append(f"{path}:{lineno}: {line.strip()}")
    return hits


def test_no_stale_split_config_refs_in_docs(repo_root: Path) -> None:
    """docs/ MUST NOT reference config/{state,action,reward}.yaml (CLAUDE.md §4)."""
    hits = _scan_for_stale_refs(repo_root / "docs")
    # Allow this test file itself to mention the stale names in STALE_REFS,
    # but docs/ should never reference them.
    assert not hits, (
        "Found stale split-config references in docs/ — replace with "
        "`config/config.yaml#<block>` notation per CLAUDE.md §4:\n"
        + "\n".join(hits)
    )
