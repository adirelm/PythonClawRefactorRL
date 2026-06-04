# UX — Nielsen Heuristics Mapping

> **Phase 4 fill — populated after the Streamlit GUI + Obsidian Graph View
> are wired to live rollout artefacts.** This file maps each of Nielsen's
> 10 usability heuristics to a concrete affordance in our two surfaces:
> the Obsidian Graph View (read-only dependency exploration) and the
> Streamlit GUI (interactive training / replay).

## Surfaces

- **Obsidian Graph View**: renders the DiGraph produced by
  `GraphifyAdapter.build(src_root, *, seed)`; used for static inspection
  of module-level coupling/cohesion before and after a refactor episode.
- **Streamlit GUI**: live PPO training dashboard + replay of saved
  trajectories from `results/traces/seed_<n>/rollout.jsonl`.

## Heuristics → affordances

| # | Heuristic | Obsidian Graph View | Streamlit GUI |
|---|---|---|---|
| 1 | Visibility of system status | <Phase 4 fill> | <Phase 4 fill> |
| 2 | Match between system and real world | <Phase 4 fill> | <Phase 4 fill> |
| 3 | User control and freedom | <Phase 4 fill> | <Phase 4 fill> |
| 4 | Consistency and standards | <Phase 4 fill> | <Phase 4 fill> |
| 5 | Error prevention | <Phase 4 fill> | <Phase 4 fill> |
| 6 | Recognition rather than recall | <Phase 4 fill> | <Phase 4 fill> |
| 7 | Flexibility and efficiency of use | <Phase 4 fill> | <Phase 4 fill> |
| 8 | Aesthetic and minimalist design | <Phase 4 fill> | <Phase 4 fill> |
| 9 | Help users recognize / diagnose / recover from errors | <Phase 4 fill> | <Phase 4 fill> |
| 10 | Help and documentation | <Phase 4 fill> | <Phase 4 fill> |

## Notes

- Screenshots and walk-through GIFs land in `docs/assets/ux/` during
  Phase 4. Each heuristic row links to its evidence asset.
- Accessibility checks (contrast, keyboard navigation) are tracked as a
  separate Phase 4 sub-deliverable and do not duplicate this table.
