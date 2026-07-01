from __future__ import annotations

import unittest

from control_runtime.orchestrators.loop_state import (
    ControlRoundSnapshot,
    OrchestratorLoopState,
    append_round_trace,
)
from control_runtime.context import (
    build_feedback_context_from_snapshots,
    build_main_context,
    build_memory_context,
)


class OrchestratorContextRefactorTest(unittest.TestCase):
    def test_loop_state_keeps_snapshots_and_derives_previous_fields(self) -> None:
        state = OrchestratorLoopState()

        for index in range(1, 5):
            append_round_trace(
                state,
                trace_payload={
                    "round_index": index,
                    "global_intent": {"round": index},
                    "operation_intent": {"op": index},
                    "policy_plan": {"plan": index},
                    "pda_feedback": {"execution_status": "Failed", "violation_details": f"failure-{index}"},
                    "diagnosis": {"root_cause_category": f"category-{index}"},
                    "negotiation_request": {"summary": f"negotiation-{index}"},
                },
                feedback_added=f"[Round Feedback]\nround_index: {index}",
            )

        self.assertEqual([snap.round_index for snap in state.rounds], [2, 3, 4])
        self.assertEqual(state.previous_diagnosis, {"root_cause_category": "category-4"})
        self.assertEqual(state.previous_report_payload["violation_details"], "failure-4")
        self.assertEqual(state.previous_negotiation_request, {"summary": "negotiation-4"})
        with self.assertRaises(Exception):
            state.rounds[-1].diagnosis = {}

    def test_feedback_context_summarizes_older_rounds_and_keeps_recent_full(self) -> None:
        snapshots = [
            ControlRoundSnapshot(
                round_index=1,
                global_intent={},
                operation_intent={},
                policy_plan={},
                diagnosis={"root_cause_category": "old_binding", "reason_summary": "wrong flow binding"},
                feedback_added="[Round Feedback]\nround_index: 1\nvery old verbose detail",
            ),
            ControlRoundSnapshot(
                round_index=2,
                global_intent={},
                operation_intent={},
                policy_plan={},
                diagnosis={"root_cause_category": "recent_policy", "reason_summary": "policy failed"},
                feedback_added="[Round Feedback]\nround_index: 2\nrecent detail",
            ),
            ControlRoundSnapshot(
                round_index=3,
                global_intent={},
                operation_intent={},
                policy_plan={},
                diagnosis={"root_cause_category": "latest_dispatch", "reason_summary": "dispatch failed"},
                feedback_added="[Round Feedback]\nround_index: 3\nlatest detail",
            ),
        ]

        context = build_feedback_context_from_snapshots(snapshots)

        self.assertIn("[Older Feedback Summary]", context)
        self.assertIn("old_binding", context)
        self.assertIn("round_index: 2", context)
        self.assertIn("round_index: 3", context)
        self.assertNotIn("very old verbose detail", context)

    def test_memory_context_prefers_entries_matching_diagnosis_and_routing_hints(self) -> None:
        class Memory:
            def retrieve(self, _: str) -> dict:
                return {
                    "short_term": [
                        {"role": "MAIN", "content": "generic context"},
                        {"role": "AD", "content": "planning_blocked optimizer preview missing"},
                    ],
                    "long_term": [
                        "general success case",
                        "optimization_strategy target_stable retry fixed planner gap",
                    ],
                }

        context = build_memory_context(
            "QoS request",
            memory_manager=Memory(),
            diagnosis_hint="planning_blocked",
            routing_hint="optimization_strategy target_stable",
        )

        self.assertLess(context.index("planning_blocked"), context.index("generic context"))
        self.assertIn("optimization_strategy target_stable", context)

    def test_main_context_is_markdown_sections_not_json_blob(self) -> None:
        import control_runtime.context.control_loop as context_api

        original = context_api.get_snapshot_data_by_id
        context_api.get_snapshot_data_by_id = lambda _: {"apps": [], "slices": [], "nodes": [], "mobility": [], "flows": []}
        try:
            context = build_main_context(
                "snap-1",
                round_index=2,
                memory_context="[Memory] relevant",
                feedback_context="[Round Feedback] previous",
                previous_diagnosis={"root_cause_category": "planning_blocked"},
            )
        finally:
            context_api.get_snapshot_data_by_id = original

        self.assertTrue(context.startswith("## Snapshot Summary"))
        self.assertIn("## Previous Round Diagnosis", context)
        self.assertIn("## Retry Hints", context)
        self.assertIn("## Memory Context", context)
        self.assertNotEqual(context[:1], "{")

    def test_context_helpers_are_not_left_in_orchestrator_support_module(self) -> None:
        import importlib.util
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        self.assertIsNone(importlib.util.find_spec("control_runtime.orchestrators.main_control_support"))

        orchestrator_text = (root / "src" / "control_runtime" / "orchestrators" / "main_control_orchestrator.py").read_text(encoding="utf-8")
        self.assertNotIn("def _build_memory_context", orchestrator_text)
        self.assertNotIn("def _build_ie_context", orchestrator_text)
        self.assertNotIn("def _should_reuse_operation_intent", orchestrator_text)


if __name__ == "__main__":
    unittest.main()
