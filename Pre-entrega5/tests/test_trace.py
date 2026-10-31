import json
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from fakes import tool_call
from react_agent.runner import TurnResult
from react_agent.trace import format_turn_log, turn_to_dict, write_trace


def sample_turn() -> TurnResult:
    return TurnResult(
        thread_id="t1",
        question="¿Cuántos pedidos?",
        answer="3 pedidos",
        messages=[
            HumanMessage("¿Cuántos pedidos?"),
            tool_call("listar_pedidos", cliente_id=102),
            ToolMessage('{"cantidad": 3}', tool_call_id="call_1", name="listar_pedidos"),
            AIMessage("3 pedidos"),
        ],
    )


def test_turn_to_dict_keeps_tool_calls_and_results():
    data = turn_to_dict(sample_turn())
    assert data["thread_id"] == "t1"
    assert data["herramientas_invocadas"] == 1
    assert data["mensajes"][1]["tool_calls"] == [{"nombre": "listar_pedidos", "argumentos": {"cliente_id": 102}}]
    assert data["mensajes"][2]["herramienta"] == "listar_pedidos"


def test_log_shows_the_react_cycle_in_order():
    log = format_turn_log(sample_turn())
    positions = [log.index(s) for s in ("¿Cuántos pedidos?", "listar_pedidos(cliente_id=102)", '{"cantidad": 3}', "Respuesta: 3 pedidos")]
    assert positions == sorted(positions)


def test_write_trace_creates_json_and_log(tmp_path: Path):
    write_trace({"multi_paso": [sample_turn()]}, tmp_path)
    data = json.loads((tmp_path / "traza_ejecucion.json").read_text(encoding="utf-8"))
    assert data["multi_paso"][0]["respuesta"] == "3 pedidos"
    assert "multi_paso" in (tmp_path / "traza_ejecucion.log").read_text(encoding="utf-8")
