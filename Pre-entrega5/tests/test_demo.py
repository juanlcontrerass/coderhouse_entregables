import asyncio
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage

from fakes import ScriptedChatModel, tool_call
from run_demo import run_scenarios


def scripted_demo_llm() -> ScriptedChatModel:
    return ScriptedChatModel(
        responses=[
            # Multi-paso
            tool_call("buscar_cliente", nombre="Lucía Fernández"),
            tool_call("listar_pedidos", call_id="call_2", cliente_id=102),
            AIMessage("3 pedidos por $14.500"),
            # Memoria
            tool_call("detalle_pedido", call_id="call_3", pedido_id=1003),
            AIMessage("El último fue el 1003"),
            # Reintento
            tool_call("buscar_cliente", call_id="call_4", nombre="Lusia Fernandes"),
            tool_call("buscar_cliente", call_id="call_5", nombre="Lucía Fernández"),
            AIMessage("Asumo que es Lucía Fernández"),
            # Aclaración
            tool_call("buscar_cliente", call_id="call_6", nombre="Roberto Sánchez"),
            AIMessage("No lo encontré, ¿podés confirmar el nombre?"),
        ]
    )


def test_demo_runs_all_scenarios(tmp_path: Path):
    scenarios = asyncio.run(run_scenarios(scripted_demo_llm(), tmp_path / "c.sqlite"))
    assert list(scenarios) == ["multi_paso", "memoria", "reintento", "aclaracion"]
    assert scenarios["multi_paso"][0].tool_call_count == 2
    assert scenarios["reintento"][0].tool_call_count == 2


def test_memory_scenario_reuses_the_first_thread(tmp_path: Path):
    llm = scripted_demo_llm()
    scenarios = asyncio.run(run_scenarios(llm, tmp_path / "c.sqlite"))
    assert scenarios["memoria"][0].thread_id == scenarios["multi_paso"][0].thread_id
    humans_sent = [m for m in llm.received[3] if isinstance(m, HumanMessage)]
    assert len(humans_sent) == 2


def test_each_run_uses_new_threads(tmp_path: Path):
    db = tmp_path / "c.sqlite"
    first = asyncio.run(run_scenarios(scripted_demo_llm(), db))
    second = asyncio.run(run_scenarios(scripted_demo_llm(), db))
    assert first["multi_paso"][0].thread_id != second["multi_paso"][0].thread_id
