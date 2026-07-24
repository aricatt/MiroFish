"""端到端测试：通过 GraphitiShim 向 Neo4j 写入数据并查询。

需要 Neo4j (bolt://localhost:7687) 和 Ollama (localhost:11434) 运行。
运行: cd backend && source venv/bin/activate && python tests/test_e2e_graphiti.py
"""

import os
import sys
import time

os.environ["GRAPH_BACKEND"] = "graphiti"
os.environ["NEO4J_URI"] = "bolt://localhost:7687"
os.environ["NEO4J_USER"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = "password"
os.environ["OLLAMA_BASE_URL"] = "http://localhost:11434"
os.environ["OLLAMA_MODEL"] = "qwen2.5:7b"
os.environ["OLLAMA_EMBED_MODEL"] = "nomic-embed-text"
os.environ["OLLAMA_EMBED_DIM"] = "768"
os.environ["LLM_API_KEY"] = "ollama"

# Load modules directly, bypassing app/__init__.py Flask import
import importlib
import importlib.util
import types

_app_dir = os.path.join(os.path.dirname(__file__), "..", "app")
_app_dir = os.path.abspath(_app_dir)

if "app" not in sys.modules:
    app_pkg = types.ModuleType("app")
    app_pkg.__path__ = [_app_dir]
    app_pkg.__package__ = "app"
    sys.modules["app"] = app_pkg

for subpkg in ("app.utils", "app.graphiti_shim"):
    if subpkg not in sys.modules:
        pkg = types.ModuleType(subpkg)
        pkg.__path__ = [os.path.join(_app_dir, subpkg.split(".")[-1])]
        pkg.__package__ = subpkg
        sys.modules[subpkg] = pkg


def _load(name, rel_path):
    if name in sys.modules and hasattr(sys.modules[name], "__file__"):
        return sys.modules[name]
    path = os.path.join(_app_dir, rel_path)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_load("app.utils.logger", os.path.join("utils", "logger.py"))
_load("app.config", "config.py")
_load("app.graphiti_shim.ollama_embedder", os.path.join("graphiti_shim", "ollama_embedder.py"))
_load("app.graphiti_shim.graph_manager", os.path.join("graphiti_shim", "graph_manager.py"))
shim_mod = _load("app.graphiti_shim.shim", os.path.join("graphiti_shim", "shim.py"))

GraphitiShim = shim_mod.GraphitiShim

GRAPH_ID = "e2e-test-" + str(int(time.time()))
TEST_TEXT = (
    "张三是阿里巴巴的软件工程师。"
    "他毕业于浙江大学计算机科学专业。"
    "李四是张三的同事，在阿里巴巴担任产品经理。"
    "张三和李四一起开发了名为通义千问的AI产品。"
)


def test_create_graph():
    """1. 创建图谱"""
    shim = GraphitiShim()
    result = shim.graph.create(graph_id=GRAPH_ID, name="E2E Test Graph")
    assert result.graph_id == GRAPH_ID
    print(f"  Graph created: {result.graph_id}")
    return shim


def test_add_episode(shim):
    """2. 添加一段文本到图谱（Graphiti 会提取实体和关系）"""
    print(f"  Adding episode text ({len(TEST_TEXT)} chars)...")
    print(f"  This may take 30-60s as Ollama processes the text...")
    result = shim.graph.add(
        graph_id=GRAPH_ID,
        type="text",
        data=TEST_TEXT,
        source_description="E2E test episode",
    )
    assert result.processed is True
    assert result.uuid_
    print(f"  Episode added: {result.uuid_}")
    return result.uuid_


def test_get_nodes(shim):
    """3. 查询图谱中的所有节点"""
    response = shim.graph.node.with_raw_response.get_by_graph_id(GRAPH_ID)
    nodes = response.data
    print(f"  Found {len(nodes)} nodes")
    for node in nodes:
        print(f"    - {node.name} (labels={node.labels})")
    assert len(nodes) > 0, "Expected at least one node after adding episode"
    return nodes


def test_get_edges(shim):
    """4. 查询图谱中的所有边"""
    response = shim.graph.edge.with_raw_response.get_by_graph_id(GRAPH_ID)
    edges = response.data
    print(f"  Found {len(edges)} edges")
    for edge in edges:
        print(f"    - {edge.name}: {edge.fact}")
    return edges


def test_search(shim):
    """5. 搜索图谱"""
    print("  Searching for '张三'...")
    result = shim.graph.search(
        graph_id=GRAPH_ID,
        query="张三",
        limit=5,
        scope="edges",
    )
    print(f"  Search returned {len(result.edges)} edges")
    for edge in result.edges:
        print(f"    - {edge.name}: {edge.fact}")
    return result


def test_delete_graph(shim):
    """6. 删除图谱"""
    shim.graph.delete(graph_id=GRAPH_ID)
    print(f"  Graph deleted: {GRAPH_ID}")


if __name__ == "__main__":
    tests = [
        ("create_graph", test_create_graph),
        ("add_episode", test_add_episode),
        ("get_nodes", test_get_nodes),
        ("get_edges", test_get_edges),
        ("search", test_search),
        ("delete_graph", test_delete_graph),
    ]

    passed = 0
    failed = 0
    shim = None

    for name, test in tests:
        print(f"\n[{name}]")
        try:
            if name == "create_graph":
                shim = test()
            elif name == "delete_graph":
                test(shim)
            elif shim:
                test(shim)
            else:
                raise RuntimeError("No shim instance")
            print(f"  PASS")
            passed += 1
        except Exception as e:
            print(f"  FAIL: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
            # Don't continue if add_episode fails — subsequent tests depend on it
            if name == "add_episode":
                break

    print(f"\n{'='*40}")
    print(f"{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
