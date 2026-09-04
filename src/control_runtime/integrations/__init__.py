"""Live integrations for the OAI/free5GC testbed."""

from .testbed import TestbedClient, TestbedConfig, TestbedSimulatorAdapter


def create_live_testbed_adapter() -> TestbedSimulatorAdapter:
    return TestbedSimulatorAdapter.from_env()

__all__ = [
    "TestbedClient",
    "TestbedConfig",
    "TestbedSimulatorAdapter",
    "create_live_testbed_adapter",
    "optimizer",
    "pcf",
    "scenario",
    "storage",
    "testbed",
]
