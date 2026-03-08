from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


def load_module():
    root = Path(__file__).resolve().parents[2]
    script = root / "tools" / "orchestrate_handoffs.py"
    spec = importlib.util.spec_from_file_location("orchestrate_handoffs", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


orchestrator = load_module()


def write_report(path: Path, role: str, task_id: str, status: str, verdict: str | None = None) -> None:
    text = [
        f"# {role.title()} Report - {task_id}",
        "",
        f"- Task ID: `{task_id}`",
        f"- Role: `{role}`",
        "- Date: `2026-03-08`",
        f"- Status: `{status}`",
        "",
        "## Inputs Consulted",
        "- `docs/backlog.md`",
        "",
        "## Summary",
        "- synthetic test report",
        "",
        "## Decisions Made",
        "- none - test fixture",
        "",
        "## Files/Modules Affected (Or Expected)",
        "- `src/example.rs` - synthetic fixture",
        "",
        "## Validation / Tests",
        "- not run",
        "",
        "## Risks / Open Questions",
        "- none - test fixture",
        "",
        "## Next Handoff",
        "- To: `Reviewer`",
        "- Requested action: `test`",
        "",
    ]
    if verdict is not None:
        text.extend(
            [
                "### Final Verdict",
                f"- `{verdict}`",
                "",
                "### Handoff To Implementer Or Human",
                "- To: `Implementer`",
                "- Requested action: `test`",
                "",
            ]
        )
    path.write_text("\n".join(text))


class OrchestratorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.task_id = "TASK-012-image-store-refactor"
        self.task_dir = self.root / "docs" / "handoffs" / self.task_id
        self.task_dir.mkdir(parents=True)

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def control(self, **overrides):
        data = orchestrator.TaskControl(
            task_id=self.task_id,
            brief="Refactor image store persistence.",
            input_refs=["docs/backlog.md"],
            **overrides,
        )
        data.save(self.task_dir / "orchestrator.json")

    def state(self):
        return orchestrator.load_task_state(self.task_dir)

    def test_missing_architect_runs_architect(self):
        self.control()
        decision = orchestrator.decide_next_action(self.state())
        self.assertEqual(decision.kind, "run_role")
        self.assertEqual(decision.role, "architect")

    def test_architect_requires_human_approval(self):
        self.control()
        arch = self.task_dir / "01-architect.md"
        write_report(arch, "architect", self.task_id, "ready_for_implementer")
        decision = orchestrator.decide_next_action(self.state())
        self.assertEqual(decision.kind, "human")
        self.assertIn("awaits human approval", decision.reason)

    def test_approved_architect_runs_implementer(self):
        arch = self.task_dir / "01-architect.md"
        write_report(arch, "architect", self.task_id, "ready_for_implementer")
        self.control(approved_architect_report_mtime=arch.stat().st_mtime)
        decision = orchestrator.decide_next_action(self.state())
        self.assertEqual(decision.kind, "run_role")
        self.assertEqual(decision.role, "implementer")

    def test_ready_for_review_runs_reviewer(self):
        arch = self.task_dir / "01-architect.md"
        impl = self.task_dir / "02-implementer.md"
        write_report(arch, "architect", self.task_id, "ready_for_implementer")
        self.control(approved_architect_report_mtime=arch.stat().st_mtime)
        write_report(impl, "implementer", self.task_id, "ready_for_review")
        decision = orchestrator.decide_next_action(self.state())
        self.assertEqual(decision.kind, "run_role")
        self.assertEqual(decision.role, "reviewer")

    def test_changes_requested_routes_back_to_implementer(self):
        arch = self.task_dir / "01-architect.md"
        impl = self.task_dir / "02-implementer.md"
        rev = self.task_dir / "03-reviewer.md"
        write_report(arch, "architect", self.task_id, "ready_for_implementer")
        self.control(approved_architect_report_mtime=arch.stat().st_mtime)
        write_report(impl, "implementer", self.task_id, "ready_for_review")
        write_report(
            rev,
            "reviewer",
            self.task_id,
            "complete",
            verdict="changes_requested",
        )
        state = self.state()
        orchestrator.synchronize_reviewer_counter(state, persist=False)
        decision = orchestrator.decide_next_action(state)
        self.assertEqual(decision.kind, "run_role")
        self.assertEqual(decision.role, "implementer")

    def test_approved_reviewer_completes_task(self):
        arch = self.task_dir / "01-architect.md"
        impl = self.task_dir / "02-implementer.md"
        rev = self.task_dir / "03-reviewer.md"
        write_report(arch, "architect", self.task_id, "ready_for_implementer")
        self.control(approved_architect_report_mtime=arch.stat().st_mtime)
        write_report(impl, "implementer", self.task_id, "ready_for_review")
        write_report(
            rev,
            "reviewer",
            self.task_id,
            "complete",
            verdict="approved",
        )
        decision = orchestrator.decide_next_action(self.state())
        self.assertEqual(decision.kind, "complete")

    def test_review_loop_limit_forces_human_gate(self):
        arch = self.task_dir / "01-architect.md"
        impl = self.task_dir / "02-implementer.md"
        rev = self.task_dir / "03-reviewer.md"
        write_report(arch, "architect", self.task_id, "ready_for_implementer")
        self.control(
            approved_architect_report_mtime=arch.stat().st_mtime,
            max_changes_requested=0,
        )
        write_report(impl, "implementer", self.task_id, "ready_for_review")
        write_report(
            rev,
            "reviewer",
            self.task_id,
            "complete",
            verdict="changes_requested",
        )
        state = self.state()
        orchestrator.synchronize_reviewer_counter(state, persist=False)
        decision = orchestrator.decide_next_action(state)
        self.assertEqual(decision.kind, "human")
        self.assertIn("review loop limit exceeded", decision.reason)


if __name__ == "__main__":
    unittest.main()
