from __future__ import annotations

from .contracts import IntentBinding, IntentLifecycle, IntentStatus, MonitorAlert
from ..orchestrators.contracts import ControlRoundResult


class IntentLifecycleManager:
    def __init__(self) -> None:
        self._lifecycles: dict[str, IntentLifecycle] = {}
        self._replanned_bindings: set[IntentBinding] = set()

    @property
    def lifecycles(self) -> list[IntentLifecycle]:
        return list(self._lifecycles.values())

    def record_initial(self, result: ControlRoundResult) -> IntentLifecycle:
        bindings = (
            [
                IntentBinding(
                    supi=flow["supi"],
                    app_id=flow["app_id"],
                    flow_id=flow["flow_id"],
                )
                for flow in result.round_traces[-1]["grounding_decision"]["flows"]
            ]
            if result.completed
            else []
        )
        lifecycle = IntentLifecycle(
            intent_id=result.scope.session_id,
            bindings=bindings,
            status=IntentStatus.ACTIVE if result.completed else IntentStatus.FAILED,
        )
        self._lifecycles[lifecycle.intent_id] = lifecycle
        return lifecycle

    def observe(
        self,
        alerts: list[MonitorAlert],
        *,
        skip_intent_ids: set[str] | None = None,
    ) -> list[tuple[IntentLifecycle, MonitorAlert]]:
        active_bindings = {self._alert_binding(alert) for alert in alerts}
        self._replanned_bindings.intersection_update(active_bindings)
        skipped_bindings = {
            binding
            for intent_id in (skip_intent_ids or set())
            for binding in self._lifecycles[intent_id].bindings
        }
        reentries: list[tuple[IntentLifecycle, MonitorAlert]] = []
        for lifecycle in self._lifecycles.values():
            if lifecycle.intent_id in (skip_intent_ids or set()):
                continue
            if (
                lifecycle.status in {IntentStatus.ACTIVE, IntentStatus.DEGRADED}
                and not any(binding in active_bindings for binding in lifecycle.bindings)
            ):
                lifecycle.status = IntentStatus.SATISFIED

        scheduled_bindings: set[IntentBinding] = set()
        for alert in alerts:
            binding = self._alert_binding(alert)
            if binding in skipped_bindings:
                continue
            lifecycle = self._lifecycle_for_binding(binding)
            if lifecycle is None:
                lifecycle = IntentLifecycle(
                    intent_id=f"monitor:{binding.supi}:{binding.app_id}:{binding.flow_id}",
                    bindings=[binding],
                    status=IntentStatus.DEGRADED,
                    origin="monitor",
                )
                self._lifecycles[lifecycle.intent_id] = lifecycle
            elif lifecycle.status is IntentStatus.FAILED:
                continue
            else:
                lifecycle.status = IntentStatus.DEGRADED
            if binding in self._replanned_bindings or binding in scheduled_bindings:
                continue
            reentries.append((lifecycle, alert))
            scheduled_bindings.add(binding)
        return reentries

    def begin_replan(self, intent_id: str, alert: MonitorAlert) -> IntentLifecycle:
        lifecycle = self._lifecycles[intent_id]
        self._replanned_bindings.add(self._alert_binding(alert))
        lifecycle.status = IntentStatus.REPLANNING
        return lifecycle

    def finish_replan(self, intent_id: str, result: Any) -> IntentLifecycle:
        lifecycle = self._lifecycles[intent_id]
        lifecycle.status = IntentStatus.ACTIVE if result.completed else IntentStatus.FAILED
        return lifecycle

    @staticmethod
    def _alert_binding(alert: MonitorAlert) -> IntentBinding:
        return IntentBinding(supi=alert.supi, app_id=alert.app_id, flow_id=alert.flow_id)

    def _lifecycle_for_binding(self, binding: IntentBinding) -> IntentLifecycle | None:
        for lifecycle in reversed(self.lifecycles):
            if binding in lifecycle.bindings:
                return lifecycle
        return None
