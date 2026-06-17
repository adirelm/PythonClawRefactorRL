"""Unit tests for ``scripts/collect_phase_corpora.py`` (Phase-4 COST-3).

Verifies the JSONL emission contract that COST-4 will consume:

* Every record carries ``phase``, ``role``, ``text``.
* Phase boundaries map commits to their expected phase index.
* Missing ``docs/PROMPTS.md`` does NOT crash collection.
* Hebrew (or any non-ASCII) commit subjects round-trip through JSONL
  without UnicodeEncodeError, matching ADR-003a's "non-ASCII regime is
  the whole reason we triple-count" stance.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "collect_phase_corpora.py"
_SPEC = importlib.util.spec_from_file_location("collect_phase_corpora", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
collect_mod = importlib.util.module_from_spec(_SPEC)
sys.modules["collect_phase_corpora"] = collect_mod
_SPEC.loader.exec_module(collect_mod)


def _git(*args: str) -> subprocess.CompletedProcess:
    """Run a git command in the repo, capturing output (never raises on failure)."""
    return subprocess.run(["git", *args], cwd=_REPO_ROOT, capture_output=True, text=True, check=False)


def _has_git_history() -> bool:
    """True iff this checkout is a git repo with >=1 commit.

    Guards the history-dependent tests so a vanished ``.git`` (Google Drive sync)
    or shallow checkout SKIPS rather than fails — the corpus contract is only
    meaningful when history is present (CI uses fetch-depth: 0).
    """
    try:
        out = _git("rev-list", "--max-parents=0", "HEAD")
    except FileNotFoundError:
        return False
    return out.returncode == 0 and bool(out.stdout.strip())


_needs_history = pytest.mark.skipif(not _has_git_history(), reason="no git history in this checkout")


@pytest.fixture
def output_dir(tmp_path: Path) -> Path:
    return tmp_path / "cost"


@_needs_history
def test_emits_jsonl_with_required_fields(output_dir: Path) -> None:
    """Every emitted record carries phase + role + text (sealed COST-4 contract)."""
    rc = collect_mod.main(["--output-dir", str(output_dir), "--prompts-md", "/nonexistent"])
    assert rc == 0
    phase_files = sorted(output_dir.glob("phase_*.jsonl"))
    assert len(phase_files) >= 4, f"expected ≥4 phase files, got {len(phase_files)}"
    total = 0
    for path in phase_files:
        for line in path.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            assert "phase" in rec
            assert "role" in rec
            assert "text" in rec
            assert isinstance(rec["text"], str)
            total += 1
    assert total > 0, "at least one record should have been emitted"


@_needs_history
def test_phase_boundaries_match_commits(output_dir: Path) -> None:
    """Each phase's marker commit lands in its declared phase file.

    Resolved at RUNTIME from the 'Phase N' commit-subject convention (not
    pinned SHAs), so the contract survives history rewrites: every phase whose
    marker appears in history must collect at least one commit whose subject
    names that phase.
    """
    rc = collect_mod.main(["--output-dir", str(output_dir), "--prompts-md", "/nonexistent"])
    assert rc == 0
    for phase in (1, 2, 3):
        marker = f"phase {phase}"
        present_in_history = _git("log", "--format=%s", "HEAD").stdout.lower()
        if marker not in present_in_history:
            continue  # phase not labelled in this history — nothing to assert
        recs = [
            json.loads(ln)
            for ln in (output_dir / f"phase_{phase}.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        subjects = [r["text"].lower() for r in recs if r["role"] == "commit_subject"]
        assert any(marker in s for s in subjects), f"no 'Phase {phase}' commit in phase_{phase}.jsonl"


def test_handles_missing_prompts_md_gracefully(output_dir: Path) -> None:
    """Absent prompts file should not crash and should not add prompt records."""
    bogus = output_dir / "definitely-does-not-exist.md"
    rc = collect_mod.main(["--output-dir", str(output_dir), "--prompts-md", str(bogus)])
    assert rc == 0
    for path in output_dir.glob("phase_*.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            assert rec["role"] != "prompts_md", "no prompts_md row should appear when file missing"


def test_unicode_safe(output_dir: Path, tmp_path: Path) -> None:
    """Hebrew text in a PROMPTS.md is emitted intact (no escape, no crash)."""
    hebrew = "שלום עולם — verbatim prompt log\n"
    prompts = tmp_path / "PROMPTS.md"
    prompts.write_text(hebrew, encoding="utf-8")
    rc = collect_mod.main(["--output-dir", str(output_dir), "--prompts-md", str(prompts)])
    assert rc == 0
    found = False
    for path in output_dir.glob("phase_*.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            if rec["role"] == "prompts_md":
                assert "שלום" in rec["text"]
                found = True
    assert found, "PROMPTS.md row missing despite file existing"


def test_record_as_jsonl_round_trips() -> None:
    """Direct check of the Record dataclass -> JSONL serializer."""
    rec = collect_mod.Record(
        phase=2,
        source="git",
        role="commit_subject",
        text="Phase 2 — custom RL env",
        timestamp="2026-06-04T01:32:12+03:00",
        sha="ec1288a",
    )
    parsed = json.loads(rec.as_jsonl())
    assert parsed["phase"] == 2
    assert parsed["role"] == "commit_subject"
    assert parsed["sha"] == "ec1288a"


@_needs_history
def test_git_range_override(output_dir: Path) -> None:
    """--git-range overrides phase-4 range; phase 4 file still emitted.

    The range boundary is resolved at runtime (root commit .. HEAD) rather than
    a hardcoded SHA, so it survives a rewritten history.
    """
    root = _git("rev-list", "--max-parents=0", "HEAD").stdout.split()[0]
    rc = collect_mod.main(
        [
            "--output-dir",
            str(output_dir),
            "--prompts-md",
            "/nonexistent",
            "--git-range",
            f"{root}..HEAD",
        ]
    )
    assert rc == 0
    p4 = output_dir / "phase_4.jsonl"
    assert p4.exists()
    recs = [json.loads(ln) for ln in p4.read_text(encoding="utf-8").splitlines()]
    assert any(r["source"] == "git" for r in recs)
