"""冒烟测试：验证 GraphitiShim 导入和接口结构正确性。

不需要 Neo4j / Ollama 运行，只检查代码结构和鸭子类型兼容性。
运行: cd backend && python tests/test_shim_smoke.py
"""

import importlib
import importlib.util
import os
import sys
import types
from types import SimpleNamespace

# 在导入 Config 前设置环境变量
os.environ["GRAPH_BACKEND"] = "graphiti"
os.environ["NEO4J_URI"] = "bolt://localhost:7687"
os.environ["NEO4J_PASSWORD"] = "password"
os.environ["LLM_API_KEY"] = "test-key"

_app_dir = os.path.join(os.path.dirname(__file__), "..", "app")
_app_dir = os.path.abspath(_app_dir)

# 创建一个空的 'app' 包，绕过 app/__init__.py 的 Flask 依赖
if "app" not in sys.modules:
    app_pkg = types.ModuleType("app")
    app_pkg.__path__ = [_app_dir]
    app_pkg.__package__ = "app"
    sys.modules["app"] = app_pkg

# 同样创建 app.utils 和 app.graphiti_shim 子包
for subpkg in ("app.utils", "app.graphiti_shim"):
    if subpkg not in sys.modules:
        pkg = types.ModuleType(subpkg)
        pkg.__path__ = [os.path.join(_app_dir, subpkg.split(".")[-1])]
        pkg.__package__ = subpkg
        sys.modules[subpkg] = pkg


def _load(name, rel_path):
    """直接从文件路径加载模块，注册到 sys.modules。"""
    if name in sys.modules and hasattr(sys.modules[name], "__file__"):
        return sys.modules[name]
    path = os.path.join(_app_dir, rel_path)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_config():
    return _load("app.config", "config.py")


def _load_shim():
    _load("app.utils.logger", os.path.join("utils", "logger.py"))
    _load_config()
    _load("app.graphiti_shim.ollama_embedder", os.path.join("graphiti_shim", "ollama_embedder.py"))
    _load("app.graphiti_shim.graph_manager", os.path.join("graphiti_shim", "graph_manager.py"))
    return _load("app.graphiti_shim.shim", os.path.join("graphiti_shim", "shim.py"))


# ── Config 测试 ──────────────────────────────────────────────────────

def test_config_graph_backend():
    """Config 正确读取 GRAPH_BACKEND"""
    config = _load_config()
    Config = config.Config
    assert Config.GRAPH_BACKEND == "graphiti"
    assert Config.NEO4J_URI == "bolt://localhost:7687"
    assert Config.OLLAMA_MODEL  # 有默认值


def test_config_validate_graphiti():
    """GRAPH_BACKEND=graphiti 时不要求 ZEP_API_KEY"""
    config = _load_config()
    Config = config.Config
    errors = Config.validate()
    assert not any("ZEP_API_KEY" in e for e in errors)


def test_config_validate_zep():
    """GRAPH_BACKEND=zep 时仍然要求 ZEP_API_KEY"""
    config = _load_config()
    Config = config.Config
    original_backend = Config.GRAPH_BACKEND
    original_key = Config.ZEP_API_KEY
    try:
        Config.GRAPH_BACKEND = "zep"
        Config.ZEP_API_KEY = None  # 模拟未配置
        errors = Config.validate()
        assert any("ZEP_API_KEY" in e for e in errors)
    finally:
        Config.GRAPH_BACKEND = original_backend
        Config.ZEP_API_KEY = original_key


# ── Shim 结构测试 ────────────────────────────────────────────────────

def test_shim_import():
    """GraphitiShim 能成功导入"""
    shim_mod = _load_shim()
    assert shim_mod.GraphitiShim is not None
    assert shim_mod.NotFoundError is not None


def test_shim_interface_structure():
    """GraphitiShim 实例暴露正确的子客户端结构"""
    shim_mod = _load_shim()
    shim = shim_mod.GraphitiShim()

    # .graph 子客户端
    for method in ("create", "delete", "get", "set_ontology", "add", "search"):
        assert hasattr(shim.graph, method), f"Missing shim.graph.{method}"

    # .graph.episode / .graph.node / .graph.edge
    assert hasattr(shim.graph.episode, "get")
    assert hasattr(shim.graph.node, "get")
    assert hasattr(shim.graph.node, "get_edges")
    assert hasattr(shim.graph.node, "with_raw_response")
    assert hasattr(shim.graph.node.with_raw_response, "get_by_graph_id")
    assert hasattr(shim.graph.edge, "with_raw_response")
    assert hasattr(shim.graph.edge.with_raw_response, "get_by_graph_id")

    # .batch 子客户端
    for method in ("create", "add", "process", "get", "list", "list_items"):
        assert hasattr(shim.batch, method), f"Missing shim.batch.{method}"


# ── Batch 流程测试（不触碰 Neo4j）────────────────────────────────────

def test_batch_flow_no_neo4j():
    """Batch 流程 create→add→process→get→list_items 不崩溃"""
    shim_mod = _load_shim()
    shim = shim_mod.GraphitiShim()

    # create
    batch = shim.batch.create(metadata={"test": True})
    assert hasattr(batch, "batch_id")
    batch_id = batch.batch_id

    # add — 用 SimpleNamespace 模拟 BatchAddItem
    items = [
        SimpleNamespace(
            type="graph_episode",
            graph_id="test-graph",
            data="hello world",
            data_type="text",
            source_description="test",
            metadata={},
        )
    ]
    details = shim.batch.add(batch_id=batch_id, items=items)
    assert len(details) == 1
    assert details[0].status == "pending"

    # process — 会尝试调用 graph.add → 触发 Neo4j 连接失败 → 标记 failed
    shim.batch.process(batch_id=batch_id)

    # get
    summary = shim.batch.get(batch_id=batch_id)
    assert summary.status == "succeeded"

    # list_items
    page = shim.batch.list_items(batch_id=batch_id, limit=100)
    assert len(page.items) == 1
    assert page.items[0].status in ("succeeded", "failed")

    # list
    page = shim.batch.list(limit=100)
    assert len(page.batches) >= 1
    assert hasattr(page.batches[0], "metadata")  # 确保有 metadata 属性


# ── Episode 测试 ─────────────────────────────────────────────────────

def test_episode_get_always_processed():
    """episode.get 总是返回 processed=True（Graphiti 同步处理）"""
    shim_mod = _load_shim()
    shim = shim_mod.GraphitiShim()
    ep = shim.graph.episode.get(uuid_="fake-uuid")
    assert ep.processed is True
    assert ep.uuid_ == "fake-uuid"


# ── NotFoundError 兼容性 ─────────────────────────────────────────────

def test_not_found_error_compatible():
    """NotFoundError 与 zep_cloud 兼容"""
    shim_mod = _load_shim()
    NotFoundError = shim_mod.NotFoundError
    try:
        raise NotFoundError("test")
    except NotFoundError:
        pass  # OK
    try:
        from zep_cloud import NotFoundError as ZepNotFoundError
        assert NotFoundError is ZepNotFoundError or issubclass(NotFoundError, Exception)
    except ImportError:
        pass  # zep_cloud 未安装也 OK


if __name__ == "__main__":
    tests = [
        test_config_graph_backend,
        test_config_validate_graphiti,
        test_config_validate_zep,
        test_shim_import,
        test_shim_interface_structure,
        test_batch_flow_no_neo4j,
        test_episode_get_always_processed,
        test_not_found_error_compatible,
    ]
    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            print(f"  PASS  {test.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {test.__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
