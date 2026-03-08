#!/usr/bin/env python3
"""Bounded orchestrator for Architect -> Implementer -> Reviewer handoffs."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


TASK_ID_RE = re.compile(r"^TASK-\d+-[a-z0-9][a-z0-9-]*$")
TOP_FIELD_RE = re.compile(r"^\s*-\s*([A-Za-z /]+):\s*(.+?)\s*$", re.MULTILINE)
SECTION_RE = re.compile(r"^###\s+(.+?)\s*$", re.MULTILINE)

ROLE_TO_FILENAME = {
    "architect": "01-architect.md",
    "implementer": "02-implementer.md",
    "reviewer": "03-reviewer.md",
}
ALLOWED_REVIEWER_VERDICTS = (
    "approved_with_minor_followups",
    "changes_requested",
    "approved",
)
NOW = lambda: datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass
class TaskControl:
    task_id: str
    brief: str
    input_refs: list[str] = field(default_factory=list)
    architect_approval_required: bool = True
    approved_architect_report_mtime: Optional[float] = None
    max_changes_requested: int = 2
    changes_requested_count: int = 0
    last_seen_reviewer_report_mtime: Optional[float] = None
    history: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def load(cls, path: Path) -> "TaskControl":
        data = json.loads(path.read_text())
        return cls(
            task_id=data["task_id"],
            brief=data.get("brief", ""),
            input_refs=list(data.get("input_refs", [])),
            architect_approval_required=bool(
                data.get("architect_approval_required", True)
            ),
            approved_architect_report_mtime=data.get("approved_architect_report_mtime"),
            max_changes_requested=int(data.get("max_changes_requested", 2)),
            changes_requested_count=int(data.get("changes_requested_count", 0)),
            last_seen_reviewer_report_mtime=data.get(
                "last_seen_reviewer_report_mtime"
            ),
            history=list(data.get("history", [])),
        )

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2, sort_keys=True) + "\n")

    def is_architect_approved(self, architect_mtime: float) -> bool:
        if not self.architect_approval_required:
            return True
        if self.approved_architect_report_mtime is None:
            return False
        return abs(self.approved_architect_report_mtime - architect_mtime) < 1e-6


@dataclass
class ReportInfo:
    role: str
    path: Path
    exists: bool
    task_id: Optional[str] = None
    status: Optional[str] = None
    verdict: Optional[str] = None
    mtime: float = 0.0
    text: str = ""


@dataclass
class Decision:
    kind: str
    task_id: str
    role: Optional[str]
    reason: str
    task_dir: Path

    @property
    def machine_actionable(self) -> bool:
        return self.kind == "run_role"


@dataclass
class TaskState:
    task_id: str
    task_dir: Path
    control_path: Path
    control: Optional[TaskControl]
    reports: dict[str, ReportInfo]
    pending_changes_requested_count: int = 0

    @property
    def architect(self) -> ReportInfo:
        return self.reports["architect"]

    @property
    def implementer(self) -> ReportInfo:
        return self.reports["implementer"]

    @property
    def reviewer(self) -> ReportInfo:
        return self.reports["reviewer"]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def handoffs_root(root: Path) -> Path:
    return root / "docs" / "handoffs"


def task_control_path(task_dir: Path) -> Path:
    return task_dir / "orchestrator.json"


def validate_task_id(task_id: str) -> None:
    if not TASK_ID_RE.match(task_id):
        raise ValueError(
            f"invalid task id {task_id!r}; expected TASK-<number>-<slug>"
        )


def strip_ticks(value: str) -> str:
    value = value.strip()
    if value.startswith("`") and value.endswith("`") and len(value) >= 2:
        return value[1:-1].strip()
    return value


def parse_top_fields(text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for key, value in TOP_FIELD_RE.findall(text):
        normalized = key.strip().lower().replace(" ", "_").replace("/", "_")
        fields[normalized] = strip_ticks(value)
    return fields


def section_body(text: str, title: str) -> str:
    matches = list(SECTION_RE.finditer(text))
    for idx, match in enumerate(matches):
        if match.group(1).strip().lower() != title.lower():
            continue
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        return text[start:end].strip()
    return ""


def parse_verdict(text: str) -> Optional[str]:
    block = section_body(text, "Final Verdict")
    if not block:
        return None
    for verdict in ALLOWED_REVIEWER_VERDICTS:
        if re.search(rf"\b{re.escape(verdict)}\b", block):
            return verdict
    return None


def load_report(role: str, path: Path) -> ReportInfo:
    if not path.exists():
        return ReportInfo(role=role, path=path, exists=False)
    text = path.read_text()
    fields = parse_top_fields(text)
    return ReportInfo(
        role=role,
        path=path,
        exists=True,
        task_id=fields.get("task_id"),
        status=fields.get("status"),
        verdict=parse_verdict(text) if role == "reviewer" else None,
        mtime=path.stat().st_mtime,
        text=text,
    )


def load_task_state(task_dir: Path) -> TaskState:
    task_id = task_dir.name
    control_path = task_control_path(task_dir)
    control = TaskControl.load(control_path) if control_path.exists() else None
    reports = {
        role: load_report(role, task_dir / filename)
        for role, filename in ROLE_TO_FILENAME.items()
    }
    state = TaskState(
        task_id=task_id,
        task_dir=task_dir,
        control_path=control_path,
        control=control,
        reports=reports,
    )
    synchronize_reviewer_counter(state, persist=False)
    return state


def synchronize_reviewer_counter(state: TaskState, persist: bool) -> bool:
    control = state.control
    reviewer = state.reviewer
    if control is None or not reviewer.exists:
        state.pending_changes_requested_count = 0 if control is None else control.changes_requested_count
        return False

    count = control.changes_requested_count
    last_seen = control.last_seen_reviewer_report_mtime
    changed = False
    if last_seen is None or reviewer.mtime > last_seen + 1e-6:
        control.last_seen_reviewer_report_mtime = reviewer.mtime
        if reviewer.verdict == "changes_requested":
            count += 1
            control.changes_requested_count = count
        changed = True
    state.pending_changes_requested_count = count
    if changed and persist:
        control.save(state.control_path)
    return changed


def record_history(control: TaskControl, role: str, result: str, reason: str) -> None:
    control.history.append(
        {
            "at": NOW(),
            "role": role,
            "result": result,
            "reason": reason,
        }
    )
    if len(control.history) > 20:
        del control.history[:-20]


def report_mismatch(report: ReportInfo, task_id: str) -> bool:
    return report.exists and report.task_id not in (None, task_id)


def decide_next_action(state: TaskState) -> Decision:
    task_id = state.task_id
    arch = state.architect
    impl = state.implementer
    rev = state.reviewer
    control = state.control

    for report in (arch, impl, rev):
        if report_mismatch(report, task_id):
            return Decision(
                kind="human",
                task_id=task_id,
                role=None,
                reason=f"{report.path.name} task id does not match {task_id}",
                task_dir=state.task_dir,
            )

    if control is not None and control.task_id != task_id:
        return Decision(
            kind="human",
            task_id=task_id,
            role=None,
            reason="orchestrator.json task_id does not match directory name",
            task_dir=state.task_dir,
        )

    if not arch.exists:
        if control is None:
            return Decision(
                kind="human",
                task_id=task_id,
                role=None,
                reason="missing architect report and orchestrator.json task seed",
                task_dir=state.task_dir,
            )
        return Decision(
            kind="run_role",
            task_id=task_id,
            role="architect",
            reason="architect report is missing",
            task_dir=state.task_dir,
        )

    if arch.status in {"blocked", "needs_human_decision"}:
        return Decision(
            kind="human",
            task_id=task_id,
            role=None,
            reason=f"architect status is {arch.status}",
            task_dir=state.task_dir,
        )
    if arch.status != "ready_for_implementer":
        return Decision(
            kind="human",
            task_id=task_id,
            role=None,
            reason=f"architect status {arch.status!r} is not actionable",
            task_dir=state.task_dir,
        )
    if control is not None and not control.is_architect_approved(arch.mtime):
        return Decision(
            kind="human",
            task_id=task_id,
            role=None,
            reason="architect plan awaits human approval",
            task_dir=state.task_dir,
        )

    if impl.exists and impl.status == "needs_architect_clarification" and arch.mtime <= impl.mtime:
        return Decision(
            kind="run_role",
            task_id=task_id,
            role="architect",
            reason="implementer requested architect clarification",
            task_dir=state.task_dir,
        )

    if not impl.exists or arch.mtime > impl.mtime + 1e-6:
        return Decision(
            kind="run_role",
            task_id=task_id,
            role="implementer",
            reason="implementer report is missing or stale relative to architect",
            task_dir=state.task_dir,
        )

    if impl.status == "blocked":
        return Decision(
            kind="human",
            task_id=task_id,
            role=None,
            reason="implementer is blocked",
            task_dir=state.task_dir,
        )
    if impl.status == "needs_architect_clarification":
        return Decision(
            kind="run_role",
            task_id=task_id,
            role="architect",
            reason="implementer requested architect clarification",
            task_dir=state.task_dir,
        )
    if impl.status != "ready_for_review":
        return Decision(
            kind="human",
            task_id=task_id,
            role=None,
            reason=f"implementer status {impl.status!r} is not actionable",
            task_dir=state.task_dir,
        )

    if not rev.exists or impl.mtime > rev.mtime + 1e-6:
        return Decision(
            kind="run_role",
            task_id=task_id,
            role="reviewer",
            reason="reviewer report is missing or stale relative to implementer",
            task_dir=state.task_dir,
        )

    if rev.status == "blocked":
        return Decision(
            kind="human",
            task_id=task_id,
            role=None,
            reason="reviewer is blocked",
            task_dir=state.task_dir,
        )
    if rev.verdict is None:
        return Decision(
            kind="human",
            task_id=task_id,
            role=None,
            reason="reviewer report is missing a final verdict",
            task_dir=state.task_dir,
        )
    if rev.verdict == "changes_requested":
        if control is not None and state.pending_changes_requested_count > control.max_changes_requested:
            return Decision(
                kind="human",
                task_id=task_id,
                role=None,
                reason=(
                    "review loop limit exceeded "
                    f"({state.pending_changes_requested_count} > {control.max_changes_requested})"
                ),
                task_dir=state.task_dir,
            )
        return Decision(
            kind="run_role",
            task_id=task_id,
            role="implementer",
            reason="reviewer requested changes",
            task_dir=state.task_dir,
        )
    if rev.verdict in {"approved", "approved_with_minor_followups"}:
        return Decision(
            kind="complete",
            task_id=task_id,
            role=None,
            reason=f"reviewer verdict is {rev.verdict}",
            task_dir=state.task_dir,
        )
    return Decision(
        kind="human",
        task_id=task_id,
        role=None,
        reason=f"unexpected reviewer verdict {rev.verdict!r}",
        task_dir=state.task_dir,
    )


def skill_path(root: Path, role: str) -> Path:
    return root / ".agents" / "skills" / role / "SKILL.md"


def build_prompt(root: Path, state: TaskState, role: str) -> str:
    task_id = state.task_id
    skill = skill_path(root, role).resolve()
    handoff_rel = f"docs/handoffs/{task_id}"
    lines = [f"[${role}]({skill})", "", f"Task ID: `{task_id}`.", ""]

    if role == "architect":
        if state.control is None:
            raise ValueError("architect prompt requires orchestrator.json seed data")
        lines.append("Use the architect skill for this task.")
        lines.append("")
        if state.control.brief:
            lines.append("Task brief:")
            lines.append(state.control.brief.strip())
            lines.append("")
        if state.control.input_refs:
            lines.append("Inputs to consult:")
            for ref in state.control.input_refs:
                lines.append(f"- `{ref}`")
            lines.append("")
        lines.append(
            f"Write or update only `{handoff_rel}/{ROLE_TO_FILENAME[role]}`."
        )
    elif role == "implementer":
        lines.append("Use the implementer skill for this task.")
        lines.append(
            f"Read `{handoff_rel}/01-architect.md` before making changes."
        )
        if state.reviewer.exists and state.reviewer.verdict == "changes_requested":
            lines.append(
                f"Treat the latest `{handoff_rel}/03-reviewer.md` findings as mandatory."
            )
        lines.append(
            f"Write or update only `{handoff_rel}/{ROLE_TO_FILENAME[role]}`."
        )
    elif role == "reviewer":
        lines.append("Use the reviewer skill for this task.")
        lines.append(
            f"Review against `{handoff_rel}/01-architect.md`, "
            f"`{handoff_rel}/02-implementer.md`, and the actual code changes."
        )
        lines.append(
            f"Write or update only `{handoff_rel}/{ROLE_TO_FILENAME[role]}`."
        )
    else:
        raise ValueError(f"unsupported role {role!r}")

    lines.append("")
    lines.append(
        "Stop and state exactly what is missing if required upstream context is absent, stale, inconsistent, or insufficient."
    )
    return "\n".join(lines) + "\n"


def codex_exec_command(args: argparse.Namespace, root: Path) -> list[str]:
    cmd = [
        args.codex_path,
        "exec",
        "-",
        "-C",
        str(root),
    ]
    if args.dangerous:
        cmd.append("--dangerously-bypass-approvals-and-sandbox")
    else:
        cmd.append("--full-auto")
    if args.model:
        cmd.extend(["-m", args.model])
    return cmd


def discover_task_dirs(root: Path, selected_task_id: Optional[str]) -> list[Path]:
    handoffs = handoffs_root(root)
    if selected_task_id:
        validate_task_id(selected_task_id)
        task_dir = handoffs / selected_task_id
        return [task_dir] if task_dir.exists() else []

    task_dirs = []
    if not handoffs.exists():
        return task_dirs
    for child in handoffs.iterdir():
        if child.is_dir() and TASK_ID_RE.match(child.name):
            task_dirs.append(child)
    return sorted(task_dirs, key=lambda path: path.name)


def render_status_line(state: TaskState, decision: Decision) -> str:
    suffix = ""
    if state.control is not None:
        suffix = (
            f" changes_requested={state.pending_changes_requested_count}/"
            f"{state.control.max_changes_requested}"
        )
    role = decision.role or "-"
    return (
        f"{state.task_id} action={decision.kind} role={role} "
        f"reason={decision.reason}{suffix}"
    )


def command_init(args: argparse.Namespace, root: Path) -> int:
    validate_task_id(args.task_id)
    task_dir = handoffs_root(root) / args.task_id
    task_dir.mkdir(parents=True, exist_ok=True)
    control_path = task_control_path(task_dir)
    if control_path.exists() and not args.force:
        raise SystemExit(
            f"{control_path} already exists; use --force to overwrite it"
        )
    control = TaskControl(
        task_id=args.task_id,
        brief=args.brief.strip(),
        input_refs=list(args.input_ref),
        architect_approval_required=not args.no_architect_approval,
        max_changes_requested=args.max_changes_requested,
    )
    control.save(control_path)
    print(f"initialized {args.task_id} at {control_path}")
    return 0


def command_approve(args: argparse.Namespace, root: Path) -> int:
    task_dir = handoffs_root(root) / args.task_id
    if not task_dir.exists():
        raise SystemExit(f"task directory does not exist: {task_dir}")
    state = load_task_state(task_dir)
    if state.control is None:
        raise SystemExit(f"missing control file: {state.control_path}")
    if args.stage != "architect":
        raise SystemExit(f"unsupported approval stage: {args.stage}")
    if not state.architect.exists:
        raise SystemExit("cannot approve architect stage without 01-architect.md")
    if state.architect.status != "ready_for_implementer":
        raise SystemExit(
            "cannot approve architect stage unless status is ready_for_implementer"
        )
    state.control.approved_architect_report_mtime = state.architect.mtime
    state.control.changes_requested_count = 0
    state.control.last_seen_reviewer_report_mtime = None
    record_history(
        state.control,
        role="human",
        result="approved",
        reason="approved architect report",
    )
    state.control.save(state.control_path)
    print(f"approved architect stage for {args.task_id}")
    return 0


def load_states(root: Path, selected_task_id: Optional[str], persist_sync: bool) -> list[TaskState]:
    states = []
    for task_dir in discover_task_dirs(root, selected_task_id):
        state = load_task_state(task_dir)
        synchronize_reviewer_counter(state, persist=persist_sync)
        states.append(state)
    return states


def command_status(args: argparse.Namespace, root: Path) -> int:
    states = load_states(root, args.task_id, persist_sync=True)
    if args.task_id and not states:
        raise SystemExit(f"task directory not found for {args.task_id}")
    if args.json:
        payload = []
        for state in states:
            decision = decide_next_action(state)
            payload.append(
                {
                    "task_id": state.task_id,
                    "decision": asdict(decision),
                    "has_control": state.control is not None,
                    "changes_requested_count": state.pending_changes_requested_count,
                }
            )
        print(json.dumps(payload, indent=2))
        return 0
    for state in states:
        print(render_status_line(state, decide_next_action(state)))
    return 0


def select_machine_actionable(states: list[TaskState]) -> Optional[tuple[TaskState, Decision]]:
    for state in states:
        decision = decide_next_action(state)
        if decision.machine_actionable:
            return state, decision
    return None


def command_run(args: argparse.Namespace, root: Path) -> int:
    if args.max_steps < 1:
        raise SystemExit("--max-steps must be >= 1")

    completed_steps = 0
    while completed_steps < args.max_steps:
        states = load_states(root, args.task_id, persist_sync=True)
        if args.task_id and not states:
            raise SystemExit(f"task directory not found for {args.task_id}")

        chosen: Optional[tuple[TaskState, Decision]]
        if args.task_id:
            state = states[0]
            chosen = (state, decide_next_action(state))
        else:
            chosen = select_machine_actionable(states)
            if chosen is None:
                print("no machine-actionable tasks found")
                return 0

        state, decision = chosen
        if not decision.machine_actionable:
            print(render_status_line(state, decision))
            return 0

        prompt = build_prompt(root, state, decision.role or "")
        cmd = codex_exec_command(args, root)

        if not args.execute:
            print(render_status_line(state, decision))
            print("")
            print("planned command:")
            print(" ".join(cmd))
            print("")
            print("planned prompt:")
            print(prompt.rstrip())
            return 0

        result = subprocess.run(cmd, input=prompt, text=True)
        if state.control is not None:
            record_history(
                state.control,
                role=decision.role or "-",
                result=f"exit_{result.returncode}",
                reason=decision.reason,
            )
            state.control.save(state.control_path)
        if result.returncode != 0:
            return result.returncode
        completed_steps += 1

    print(f"completed {completed_steps} orchestrated step(s)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Bounded orchestrator for Architect -> Implementer -> Reviewer tasks."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser(
        "init", help="Initialize orchestrator control state for a task."
    )
    init_parser.add_argument("task_id")
    init_parser.add_argument("--brief", required=True)
    init_parser.add_argument("--input-ref", action="append", default=[])
    init_parser.add_argument("--max-changes-requested", type=int, default=2)
    init_parser.add_argument("--no-architect-approval", action="store_true")
    init_parser.add_argument("--force", action="store_true")

    approve_parser = subparsers.add_parser(
        "approve", help="Record a human approval gate for a task."
    )
    approve_parser.add_argument("task_id")
    approve_parser.add_argument("stage", choices=["architect"])

    status_parser = subparsers.add_parser(
        "status", help="Show current orchestrator state for one or more tasks."
    )
    status_parser.add_argument("--task-id")
    status_parser.add_argument("--json", action="store_true")

    run_parser = subparsers.add_parser(
        "run", help="Dispatch the next machine-actionable role."
    )
    run_parser.add_argument("--task-id")
    run_parser.add_argument("--execute", action="store_true")
    run_parser.add_argument("--max-steps", type=int, default=1)
    run_parser.add_argument("--codex-path", default="codex")
    run_parser.add_argument("--model")
    run_parser.add_argument("--dangerous", action="store_true")

    return parser


def main(argv: list[str]) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    root = repo_root()

    if args.command == "init":
        return command_init(args, root)
    if args.command == "approve":
        return command_approve(args, root)
    if args.command == "status":
        return command_status(args, root)
    if args.command == "run":
        return command_run(args, root)
    raise SystemExit(f"unknown command: {args.command}")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
