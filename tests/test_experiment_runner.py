from __future__ import annotations

from pathlib import Path

from experiments import ExperimentCase, PreparedEpisode, run_case
from shared.contracts.coordination import CoordinationEvent, TerminalPayload
from shared.runtime.journal import EpisodeJournal


class _Backend:
    def __init__(self, root: Path) -> None:
        self.root = root

    def prepare(self, case: ExperimentCase) -> PreparedEpisode:
        journal = EpisodeJournal(root=self.root, episode_id=case.episode_id)

        def execute() -> CoordinationEvent[object]:
            return journal.append_event(
                round_index=0,
                source_agent="episode_controller",
                target_agent="episode_controller",
                message_type="terminal",
                payload=TerminalPayload(
                    outcome="success",
                    resolution="executed",
                    reason_code="experiment_complete",
                    reason="experiment complete",
                    reward_total=1.0,
                    reward_components={"objective": 1.0},
                ),
            )

        return PreparedEpisode(episode_directory=journal.directory, execute=execute)


def test_experiment_runner_preserves_native_episode_contract(tmp_path: Path) -> None:
    case = ExperimentCase(
        case_id="case-1",
        episode_id="episode-1",
        method="multi_agent_control",
        user_input="prioritize the video flow",
    )

    result = run_case(case, backend=_Backend(tmp_path))

    assert result.case_id == case.case_id
    assert result.method == "multi_agent_control"
    assert result.terminal.outcome == "success"
    assert Path(result.episode_directory).is_dir()
    assert sorted(path.name for path in Path(result.episode_directory).iterdir()) == [
        "events.jsonl",
        "traces.jsonl",
    ]
