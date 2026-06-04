# QUALITY.md — A4 quality gates + honest framing (Phase 1 fill pending)

This document captures the V3 §13 ISO/IEC 25010 mapping and the
honest-limitations section. It will be filled out during Phase 1 alongside
the GRAPHIFY + Skills shim implementation. For now, placeholder content
pointing to where each section will live:

- Functional Suitability: see docs/PRD.md §5 acceptance criteria
- Performance Efficiency: see docs/COST_ANALYSIS.md (Phase 4 deliverable D8)
- Compatibility: Python ≥3.11 pinned; uv.lock locked
- Usability: see docs/UX.md (Phase 4 placeholder)
- Reliability: see ADR-008 fallback rule + ADR-010 convergence
- Security: secrets via .env (.env-example committed; .env gitignored)
- Maintainability: ≤150 LOC, ruff clean, ≥85% coverage, 11 ADRs
- Portability: pyproject.toml + uv.lock

## Honest limitations (V3 §1.4 architect transparency)
See docs/PRD.md §7. Self-grade target: 88-92 (NOT 100). A1 over-confidence lesson honored.
