from __future__ import annotations

import ast
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from shared.contracts import ExecutionScope


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
ARTIFACTS_PATH = SRC / "shared" / "runtime" / "storage" / "artifacts.py"
QUEUE_PATH = SRC / "shared" / "runtime" / "storage" / "queue.py"
PERSISTENCE_PATH = SRC / "shared" / "runtime" / "storage" / "persistence.py"
RUNTIME_STORE_PATH = SRC / "shared" / "runtime" / "storage" / "runtime_store.py"
WORKER_PATH = SRC / "shared" / "runtime" / "worker.py"
FACTORY_PATH = SRC / "application_runtime" / "persistence.py"

PERSISTENCE_BACKED_WORKER_CONSTRUCTORS = (
    (SRC / "control_runtime" / "agents" / "grounding" / "agent.py", "IntentEncodingAgent"),
    (SRC / "control_runtime" / "agents" / "planning" / "agent.py", "OptimizationStrategyAgent"),
    (SRC / "control_runtime" / "agents" / "dispatch" / "agent.py", "PolicyDispatchAgent"),
    (SRC / "control_runtime" / "agents" / "single" / "agent.py", "SingleControlAgent"),
    (SRC / "control_runtime" / "diagnostics" / "mediation" / "agent.py", "ConflictResolutionTool"),
    (SRC / "control_runtime" / "diagnostics" / "diagnosis" / "agent.py", "AssuranceDiagnosisTool"),
    (SRC / "environment" / "agent.py", "EnvironmentGenerationAgent"),
)


def _load_source_module(
    monkeypatch: pytest.MonkeyPatch,
    *,
    module_name: str,
    path: Path,
) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


def _package(name: str) -> ModuleType:
    module = ModuleType(name)
    module.__path__ = []  # type: ignore[attr-defined]
    return module


def _load_storage_modules(monkeypatch: pytest.MonkeyPatch) -> tuple[ModuleType, ModuleType, ModuleType, ModuleType]:
    shared = _package("shared")
    runtime = _package("shared.runtime")
    storage = _package("shared.runtime.storage")
    core = _package("shared.runtime.core")
    io = _package("shared.runtime.io")
    workspace_module = ModuleType("shared.runtime.core.workspace")
    files_module = ModuleType("shared.runtime.io.files")
    paths_module = ModuleType("shared.runtime.io.paths")

    class AgentWorkspace:
        def __init__(self, cache_dir: Path) -> None:
            self.cache_dir = cache_dir

        @classmethod
        def for_agent(cls, agent_name: str) -> "AgentWorkspace":
            return cls(Path.cwd() / ".runtime-persistence-test" / str(agent_name))

    def ensure_directory(path: Path) -> Path:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def read_json_file(path: Path) -> dict[str, Any]:
        return json.loads(Path(path).read_text(encoding="utf-8"))

    def write_json_file_atomic(path: Path, payload: dict[str, Any]) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_name(f".{path.name}.tmp")
        temporary_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        temporary_path.replace(path)
        return path

    workspace_module.AgentWorkspace = AgentWorkspace
    files_module.ensure_directory = ensure_directory
    files_module.read_json_file = read_json_file
    files_module.write_json_file_atomic = write_json_file_atomic
    paths_module.runtime_interfaces_root = lambda: Path.cwd() / ".runtime-persistence-test" / "interfaces"
    paths_module.queue_root = lambda agent_name: Path.cwd() / ".runtime-persistence-test" / "queue" / str(agent_name)

    for name, module in (
        ("shared", shared),
        ("shared.runtime", runtime),
        ("shared.runtime.storage", storage),
        ("shared.runtime.core", core),
        ("shared.runtime.io", io),
        ("shared.runtime.core.workspace", workspace_module),
        ("shared.runtime.io.files", files_module),
        ("shared.runtime.io.paths", paths_module),
    ):
        monkeypatch.setitem(sys.modules, name, module)

    core.AgentWorkspace = AgentWorkspace
    persistence = _load_source_module(
        monkeypatch,
        module_name="shared.runtime.storage.persistence",
        path=PERSISTENCE_PATH,
    )
    storage.RuntimePersistence = persistence.RuntimePersistence
    artifacts = _load_source_module(
        monkeypatch,
        module_name="shared.runtime.storage.artifacts",
        path=ARTIFACTS_PATH,
    )
    storage.ArtifactCache = artifacts.ArtifactCache
    storage.ArtifactEnvelope = artifacts.ArtifactEnvelope
    storage.ArtifactStore = artifacts.ArtifactStore
    queue = _load_source_module(
        monkeypatch,
        module_name="shared.runtime.storage.queue",
        path=QUEUE_PATH,
    )
    storage.FileTaskQueue = queue.FileTaskQueue
    worker = _load_source_module(
        monkeypatch,
        module_name="shared.runtime.worker",
        path=WORKER_PATH,
    )
    return artifacts, queue, worker, persistence


class _FakePersistence:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.record_result = True
        self.enqueue_result = True
        self.lease_result = True
        self.complete_result = True
        self.task_status = "failed"
        self.lease_path: Path | None = None

    def record_artifact(self, **kwargs: Any) -> bool:
        assert Path(kwargs["path"]).is_file(), "artifact bytes must reach disk before persistence"
        self.calls.append(("record_artifact", kwargs))
        return self.record_result

    def enqueue_task(self, **kwargs: Any) -> bool:
        self.calls.append(("enqueue_task", kwargs))
        return self.enqueue_result

    def acquire_task_lease(self, **kwargs: Any) -> bool:
        if self.lease_path is not None:
            assert not self.lease_path.exists(), "database lease must precede the local lease file"
        self.calls.append(("acquire_task_lease", kwargs))
        return self.lease_result

    def complete_task(self, **kwargs: Any) -> bool:
        if self.lease_path is not None:
            assert not self.lease_path.exists(), "local lease file must be cleared before task completion"
        self.calls.append(("complete_task", kwargs))
        return self.complete_result

    def get_task_status(self, artifact_id: str, target_agent: str) -> str | None:
        self.calls.append(("get_task_status", {"artifact_id": artifact_id, "target_agent": target_agent}))
        return self.task_status


def _call_names(fake: _FakePersistence) -> list[str]:
    return [name for name, _ in fake.calls]


def _compile_constructor(path: Path, class_name: str, namespace: dict[str, Any]) -> Any:
    module = ast.parse(path.read_text(encoding="utf-8"))
    class_node = next(node for node in module.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    init = next(node for node in class_node.body if isinstance(node, ast.FunctionDef) and node.name == "__init__")
    isolated = ast.Module(
        body=[
            ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
            init,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(isolated), str(path), "exec"), namespace)
    return namespace["__init__"]


class _ConstructedDependency:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args = args
        self.kwargs = kwargs
        self.model_name = str(kwargs.get("model_name") or "fake-model")

    def get_llm(self) -> "_ConstructedDependency":
        return self


def _envelope(
    artifacts: ModuleType,
    artifact_id: str = "artifact-1",
    target_agent: str = "policy_dispatch",
) -> Any:
    return artifacts.ArtifactEnvelope(
        artifact_id=artifact_id,
        correlation_id="corr-1",
        artifact_type="PolicyPlanDraft",
        payload_schema_name="PolicyPlanDraft",
        source_agent="main_control",
        target_agent=target_agent,
        scope=ExecutionScope(
            session_id="session-1",
            thread_id="session-1",
            snapshot_id="snapshot-1",
        ),
        payload={"policy": "draft"},
    )


def test_explicit_port_identity_and_runtime_call_order(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    artifacts, queue_module, _worker, _protocol = _load_storage_modules(monkeypatch)
    persistence = _FakePersistence()
    store = artifacts.ArtifactStore(persistence=persistence, root=tmp_path / "interfaces")
    queue = queue_module.FileTaskQueue(
        "policy_dispatch",
        persistence=persistence,
        root=tmp_path / "queue",
        artifact_store=store,
    )
    envelope = _envelope(artifacts)

    assert store.persistence is persistence
    assert queue.persistence is persistence
    assert queue.store is store

    request_path = store.write_request(envelope)
    assert request_path.is_file()
    assert _call_names(persistence) == ["record_artifact", "enqueue_task"]

    persistence.lease_path = queue._lease_path(envelope.artifact_id)
    assert queue.claim_artifact(envelope) is True
    assert persistence.lease_path.is_file()
    queue.mark_success(envelope)
    assert not persistence.lease_path.exists()

    persistence.task_status = "dead_letter"
    assert queue.claim_artifact(envelope) is True
    queue.mark_failure(envelope, RuntimeError("simulated failure"))
    assert (queue.dead_letter_dir / f"{envelope.artifact_id}.json").is_file()
    assert _call_names(persistence) == [
        "record_artifact",
        "enqueue_task",
        "acquire_task_lease",
        "complete_task",
        "acquire_task_lease",
        "complete_task",
        "get_task_status",
    ]


def test_persistence_failures_remain_visible_after_disk_write(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    artifacts, queue_module, _worker, _protocol = _load_storage_modules(monkeypatch)
    persistence = _FakePersistence()
    persistence.record_result = False
    store = artifacts.ArtifactStore(persistence=persistence, root=tmp_path / "interfaces")
    response = _envelope(
        artifacts,
        artifact_id="artifact-response",
        target_agent="main_control",
    )

    with pytest.raises(RuntimeError, match="Failed to record response artifact metadata"):
        store.write_response(response)
    assert list(store.response_dir(response.source_agent, response.target_agent).glob("*.json"))
    assert _call_names(persistence) == ["record_artifact"]

    completed_failure = _FakePersistence()
    completed_failure.complete_result = False
    queue = queue_module.FileTaskQueue(
        "policy_dispatch",
        persistence=completed_failure,
        root=tmp_path / "queue-failure",
        artifact_store=artifacts.ArtifactStore(persistence=completed_failure, root=tmp_path / "interfaces-failure"),
    )
    envelope = _envelope(artifacts, artifact_id="artifact-complete")
    completed_failure.lease_path = queue._lease_path(envelope.artifact_id)
    assert queue.claim_artifact(envelope) is True
    with pytest.raises(RuntimeError, match="Failed to mark task success"):
        queue.mark_success(envelope)


def test_worker_mixin_keeps_the_explicit_port_instance(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    artifacts, queue_module, worker, _protocol = _load_storage_modules(monkeypatch)
    monkeypatch.setattr(artifacts, "runtime_interfaces_root", lambda: tmp_path / "interfaces")
    monkeypatch.setattr(queue_module, "queue_root", lambda _agent_name: tmp_path / "queue")

    class Workspace:
        def __init__(self, cache_dir: Path) -> None:
            self.cache_dir = cache_dir

    monkeypatch.setattr(
        worker.AgentWorkspace,
        "for_agent",
        classmethod(lambda _cls, agent_name: Workspace(tmp_path / "cache" / agent_name)),
    )

    class ControlLikeWorker(worker.ArtifactWorkerMixin):
        agent_name = "control_like"

    persistence = _FakePersistence()
    control_worker = ControlLikeWorker()
    control_worker.init_worker_runtime(persistence=persistence)
    assert control_worker.runtime_persistence is persistence
    assert control_worker.artifact_store.persistence is persistence
    assert control_worker.queue.persistence is persistence
    assert control_worker.queue.store is control_worker.artifact_store

    class EnvironmentLikeWorker(worker.ArtifactWorkerMixin):
        agent_name = "environment_like"

    with pytest.raises(RuntimeError, match="explicit RuntimePersistence"):
        EnvironmentLikeWorker().ensure_worker_runtime_initialized()


def test_only_persistence_backed_workers_accept_persistence() -> None:
    for path, class_name in PERSISTENCE_BACKED_WORKER_CONSTRUCTORS:
        module = ast.parse(path.read_text(encoding="utf-8"))
        class_node = next(node for node in module.body if isinstance(node, ast.ClassDef) and node.name == class_name)
        init = next(node for node in class_node.body if isinstance(node, ast.FunctionDef) and node.name == "__init__")
        argument_names = [argument.arg for argument in (*init.args.args, *init.args.kwonlyargs)]
        assert "persistence" in argument_names, path
        runtime_calls = [
            node
            for node in ast.walk(init)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in {"initialize_agent_runtime", "init_worker_runtime"}
        ]
        assert runtime_calls, path
        assert all(any(keyword.arg == "persistence" for keyword in call.keywords) for call in runtime_calls), path

    main_module = ast.parse((SRC / "control_runtime" / "agents" / "main" / "agent.py").read_text(encoding="utf-8"))
    main_class = next(node for node in main_module.body if isinstance(node, ast.ClassDef) and node.name == "MainControlAgent")
    main_init = next(node for node in main_class.body if isinstance(node, ast.FunctionDef) and node.name == "__init__")
    main_arguments = [argument.arg for argument in (*main_init.args.args, *main_init.args.kwonlyargs)]
    assert "persistence" not in main_arguments

    for path in (
        SRC / "control_runtime" / "orchestrators" / "main_control_orchestrator.py",
        SRC / "control_runtime" / "orchestrators" / "single_agent_orchestrator.py",
    ):
        source = path.read_text(encoding="utf-8")
        assert "_configure_runtime_store" not in source
        assert "create_runtime_persistence" in source
        assert "persistence=self.runtime_persistence" in source


def test_main_and_single_composition_fan_out_one_explicit_instance() -> None:
    persistence = object()
    main_namespace: dict[str, Any] = {
        "MainControlAgent": _ConstructedDependency,
        "IntentEncodingAgent": _ConstructedDependency,
        "OptimizationStrategyAgent": _ConstructedDependency,
        "PolicyDispatchAgent": _ConstructedDependency,
        "ConflictResolutionTool": _ConstructedDependency,
        "AssuranceDiagnosisTool": _ConstructedDependency,
        "MemoryManager": _ConstructedDependency,
        "TokenCounter": _ConstructedDependency,
        "ContextPolicy": _ConstructedDependency,
        "create_runtime_persistence": lambda: pytest.fail("explicit port should avoid the default factory"),
    }
    main_init = _compile_constructor(
        SRC / "control_runtime" / "orchestrators" / "main_control_orchestrator.py",
        "MainControlOrchestrator",
        main_namespace,
    )
    main_harness = type("MainHarness", (), {"_DEEPSEEK_MODEL": "fake-model", "__init__": main_init})
    main = main_harness(persistence=persistence, preload_models=False)
    assert main.runtime_persistence is persistence
    assert "persistence" not in main.main_agent.kwargs
    for child in (main.ie_agent, main.os_agent, main.pd_agent, main.cr_tool, main.ad_tool):
        assert child.kwargs["persistence"] is persistence

    single_namespace: dict[str, Any] = {
        "SingleControlAgent": _ConstructedDependency,
        "PolicyDispatchAgent": _ConstructedDependency,
        "ConflictResolutionTool": _ConstructedDependency,
        "AssuranceDiagnosisTool": _ConstructedDependency,
        "create_runtime_persistence": lambda: pytest.fail("explicit port should avoid the default factory"),
    }
    single_init = _compile_constructor(
        SRC / "control_runtime" / "orchestrators" / "single_agent_orchestrator.py",
        "SingleAgentOrchestrator",
        single_namespace,
    )
    single_harness = type("SingleHarness", (), {"__init__": single_init})
    single_namespace["SingleAgentOrchestrator"] = single_harness
    single = single_harness(persistence=persistence)
    assert single.runtime_persistence is persistence
    for child in (single.single_agent, single.pd_agent, single.cr_tool, single.ad_tool):
        assert child.kwargs["persistence"] is persistence


def test_runtime_persistence_has_no_global_configuration_or_application_imports() -> None:
    runtime_store_source = RUNTIME_STORE_PATH.read_text(encoding="utf-8")
    store_tree = ast.parse(runtime_store_source)
    assert "_session_factory" not in runtime_store_source
    assert "_orm" not in runtime_store_source
    assert not any(isinstance(node, ast.Global) for node in ast.walk(store_tree))
    assert not any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "configure"
        for node in store_tree.body
    )
    persistence_class = next(
        node
        for node in store_tree.body
        if isinstance(node, ast.ClassDef) and node.name == "SqlAlchemyRuntimePersistence"
    )
    init = next(node for node in persistence_class.body if isinstance(node, ast.FunctionDef) and node.name == "__init__")
    assigned_attributes = {
        target.attr
        for node in ast.walk(init)
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == "self"
    }
    assert {"session_factory", "orm"} <= assigned_attributes
    assert any(
        isinstance(node, ast.FunctionDef) and node.name == "session_scope"
        for node in persistence_class.body
    )

    factory_source = FACTORY_PATH.read_text(encoding="utf-8")
    assert "def create_runtime_persistence" in factory_source
    factory_tree = ast.parse(factory_source)
    assert not any(
        isinstance(node, ast.ImportFrom) and (node.module or "").startswith("database")
        for node in factory_tree.body
    )
    for root in (SRC / "shared" / "runtime", SRC / "shared" / "agents"):
        for path in root.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            imported_modules = [
                node.module or ""
                for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom)
            ] + [
                alias.name
                for node in ast.walk(tree)
                if isinstance(node, ast.Import)
                for alias in node.names
            ]
            assert not any(
                module == "database"
                or module.startswith("database.")
                or module == "control_runtime"
                or module.startswith("control_runtime.")
                for module in imported_modules
            ), path


def test_orchestrator_modules_import_without_configuring_database() -> None:
    environment = os.environ.copy()
    environment.pop("DATABASE_URL", None)
    environment["PYTHONPATH"] = os.pathsep.join((str(ROOT), str(SRC)))
    command = (
        "import sys; "
        "import application_runtime.persistence; "
        "import control_runtime.orchestrators.main_control_orchestrator; "
        "import control_runtime.orchestrators.single_agent_orchestrator; "
        "assert 'database.connection' not in sys.modules"
    )

    completed = subprocess.run(
        [sys.executable, "-c", command],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
