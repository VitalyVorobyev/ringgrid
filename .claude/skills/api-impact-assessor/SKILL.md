---
name: api-impact-assessor
description: Assess integration impact of an algorithm change on config/types/API and define a migration path with minimal surface area.
---

## When to use
After an Algorithm Improvement Designer proposes a change and you need to integrate it into the system safely.

## Prerequisites
- Design document + ADR draft exist
- Task exists in `state/backlog.md`

## Inputs to consult
- Design document + ADR draft
- Current config structs, public types, and pipeline wiring
- Existing toggles/strategy enums (e.g., `CircleRefinementMethod`)

## Outputs (deliverables)
- **Integration design note** (how it fits + what changes)
- **Handoff note → Algorithm Engineer** with constraints and decisions

## Decision checklist
1. **Config knobs needed?** Prefer in this order:
   - New field on an existing config struct (**preferred**)
   - New enum variant (extend strategy enum)
   - New config struct (**last resort**)
2. **New public types required?** Avoid if possible; keep internal.
3. **Can old and new coexist behind a toggle for validation?**
4. **Migration path**
   - toggle → validate → remove old (default)
   - or keep old as fallback (only with clear use cases)

## Integration constraints to define
- Where the toggle lives (config path + defaults)
- Compatibility rules (what existing configs do by default)
- What downstream modules must remain unchanged
- What constitutes “safe to remove old path”

## Handoff note template (to Algorithm Engineer)
- Approved config shape (field/enum/struct) + exact naming
- Toggle policy and default
- Any public API constraints (must not change / allowed changes)
- Required docs updates (if any)
- Acceptance criteria additions you require for decision-making
