# Planning Workflow

Use this workflow for: discussing priorities, planning milestones, triaging new work, reviewing progress, or deciding what to tackle next.

This is the only workflow that involves direct human conversation. All other workflows are dispatched from here.

## Phases

### 1. Discuss (Project Lead + Human)

**Goal:** Align on what matters and what to work on.

**Steps:**
1. Review current `state/backlog.md`:
   - What's in Active Sprint? Any blockers?
   - What's Up Next? Still the right priority order?
   - Any completed tasks to acknowledge?
2. Discuss new ideas, reported issues, or goals from the human
3. For each new item, assess:
   - Type: feature / bug / perf / algo / infra
   - Priority: P0 (blocking) / P1 (next) / P2 (planned) / P3 (someday)
   - Scope: which pipeline stages, which modules, how many roles involved?
4. Agree on what to spec and dispatch

### 2. Specify (Project Lead)

**Goal:** Turn agreed-upon work into actionable task specs.

**Steps:**
1. For each agreed item, fill in `templates/task-spec.md`:
   - Problem statement (what and why)
   - Affected pipeline stages (check the 10-stage list)
   - Affected modules (file paths under `crates/ringgrid/src/`)
   - Public API impact assessment
   - Acceptance criteria with measurable thresholds
   - Accuracy and performance constraints
   - Python tooling changes (if any)
2. Assign an ID: `[FEAT|BUG|PERF|ALGO|INFRA]-NNN`
3. Save task spec to `state/sessions/YYYY-MM-DD-TASKID-spec.md`

### 3. Dispatch (Project Lead)

**Goal:** Kick off the right workflow with the right starting role.

**Steps:**
1. Choose workflow based on task type:
   | Type | Workflow | Starting Role |
   |------|----------|--------------|
   | feature | `feature-development.md` | Pipeline Architect |
   | bug | `bug-fix.md` | Algorithm Engineer |
   | perf | `performance-optimization.md` | Performance Engineer |
   | algo | `algorithm-improvement.md` | Algorithm Engineer |
2. Write initial handoff note from `templates/handoff-note.md`:
   - Source role: Project Lead
   - Target role: starting role for the chosen workflow
   - Include link to task spec
   - Include recommended first steps
3. Save to `state/sessions/YYYY-MM-DD-TASKID-lead-[target].md`
4. Update `state/backlog.md`:
   - Add task to Active Sprint
   - Set status to `in-progress`
   - Note the active role

### 4. Track (Project Lead)

**Goal:** Monitor progress and close completed work.

**Steps:**
1. Read session notes in `state/sessions/` for active tasks
2. Update `state/backlog.md` status as roles complete phases
3. When final handoff returns to Project Lead:
   - Verify all acceptance criteria from task spec are met
   - Move task to Done in backlog with completion date
   - Summarize outcome for the human
4. Flag any blocked or stalled tasks for human attention
