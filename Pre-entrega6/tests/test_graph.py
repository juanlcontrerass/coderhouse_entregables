"""El grafo real con un LLM falso por rol: delegación, contexto acotado y paradas."""

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.vectorstores import InMemoryVectorStore

from config import MAX_REVISIONS, MAX_VALIDATION_RETRIES
from graph import build_graph
from runner import run_query
from tests.fakes import ScriptedChatModel, decide, tool_call
from tools.retrieval import load_documents, set_vectorstore, split_documents


@pytest.fixture(autouse=True)
def vacation_store() -> None:
    """Los embeddings falsos no tienen semántica: se deja solo la política de vacaciones
    para que la fuente recuperada sea predecible."""
    docs = [d for d in load_documents() if d.metadata["source"] == "politica_vacaciones.md"]
    set_vectorstore(InMemoryVectorStore.from_documents(split_documents(docs), DeterministicFakeEmbedding(size=64)))

QUERY = "¿Cuántos días de vacaciones corresponden con 3, 7 y 12 años? Calcula el promedio."
RESEARCH_FINDING = "3 años: 14 días; 7 años: 21 días; 12 años: 28 días [Fuente: politica_vacaciones.md]"


def _models(supervisor: list[AIMessage], synthesizer: list[AIMessage]) -> dict[str, ScriptedChatModel]:
    return {
        "supervisor": ScriptedChatModel(responses=supervisor),
        "researcher": ScriptedChatModel(
            responses=[tool_call("buscar_documentos", consulta="vacaciones antigüedad"), AIMessage(RESEARCH_FINDING)]
        ),
        "analyst": ScriptedChatModel(
            responses=[tool_call("estadisticas", valores=[14, 21, 28]), AIMessage("Promedio: 21 días")]
        ),
        "synthesizer": ScriptedChatModel(responses=synthesizer),
    }


def _graph(models: dict[str, ScriptedChatModel]):  # type: ignore[no-untyped-def]
    return build_graph(
        models["supervisor"],
        researcher_llm=models["researcher"],
        analyst_llm=models["analyst"],
        synthesizer_llm=models["synthesizer"],
    )


HAPPY_SUPERVISOR = [
    decide("researcher", "Busca los días de vacaciones por antigüedad"),
    decide("analyst", "Calcula el promedio de 14, 21 y 28"),
    decide("synthesizer", "Responde al usuario"),
]
GOOD_ANSWER = AIMessage("El promedio es 21 días [Fuente: politica_vacaciones.md]")


async def test_supervisor_delegates_researcher_then_analyst_then_closes() -> None:
    models = _models(HAPPY_SUPERVISOR, [GOOD_ANSWER])
    visited: list[str] = []
    state = await run_query(_graph(models), QUERY, on_event=lambda node, _: visited.append(node))

    assert visited == ["supervisor", "researcher", "supervisor", "analyst", "supervisor", "synthesizer", "validator"]
    assert state["task_completed"] and state["final_answer"] == GOOD_ANSWER.content
    assert [c["agent"] for c in state["contributions"]] == ["researcher", "analyst", "synthesizer", "validator"]
    research = state["contributions"][0]
    assert research["sources"] == ["politica_vacaciones.md"]  # salen del ToolMessage, no del LLM
    assert research["instruction"] == "Busca los días de vacaciones por antigüedad"
    assert state["step_count"] == 3 and not any(d["forced"] for d in state["decisions"])


async def test_specialists_receive_only_their_task_not_the_whole_history() -> None:
    models = _models(HAPPY_SUPERVISOR, [GOOD_ANSWER])
    await run_query(_graph(models), QUERY)

    researcher_first_call = models["researcher"].received[0]
    assert [type(m) for m in researcher_first_call] == [SystemMessage, HumanMessage]
    assert researcher_first_call[1].content == "Busca los días de vacaciones por antigüedad"

    analyst_first_call = models["analyst"].received[0]
    assert [type(m) for m in analyst_first_call] == [SystemMessage, HumanMessage]
    task = str(analyst_first_call[1].content)
    assert "Calcula el promedio de 14, 21 y 28" in task and RESEARCH_FINDING in task
    # ni la consulta original, ni los fragmentos crudos de la búsqueda, ni los mensajes del supervisor
    assert QUERY not in task and "---" not in task and "[→" not in task

    # el supervisor tampoco ve las llamadas a herramientas internas de los agentes
    supervisor_last_call = models["supervisor"].received[-1]
    assert not any(isinstance(m, ToolMessage) for m in supervisor_last_call)
    assert models["researcher"].bound_tools == ["buscar_documentos", "listar_documentos"]
    assert "analizar_sentimiento" in models["analyst"].bound_tools


async def test_endless_redelegation_is_cut_by_the_revision_guard() -> None:
    # Un supervisor "indeciso" que siempre pide más investigación
    models = _models([decide("researcher", "Busca más")], [AIMessage("Respuesta parcial [Fuente: politica_vacaciones.md]")])
    state = await run_query(_graph(models), QUERY)

    route = [d["next_agent"] for d in state["decisions"]]
    assert route == ["researcher"] * (1 + MAX_REVISIONS) + ["synthesizer"]
    assert state["decisions"][-1]["forced"]
    assert state["revisions"] == {"researcher": MAX_REVISIONS}
    assert state["task_completed"]


async def test_validator_sends_back_invented_sources_and_ends_with_warning() -> None:
    models = _models(HAPPY_SUPERVISOR + [decide("synthesizer", "Corrige las fuentes")], [AIMessage("21 días [Fuente: inventada.md]")])
    visited: list[str] = []
    state = await run_query(_graph(models), QUERY, on_event=lambda node, _: visited.append(node))

    assert visited.count("validator") == 1 + MAX_VALIDATION_RETRIES
    assert visited[-3:] == ["supervisor", "synthesizer", "validator"]
    assert state["task_completed"]
    assert "Aviso de validación" in state["final_answer"] and "inventada.md" in state["final_answer"]
    # el sintetizador recibió el feedback del validador en su segundo intento
    assert "inventada.md" in str(models["synthesizer"].received[-1][-1].content)


async def test_diagram_contains_every_node() -> None:
    mermaid = _graph(_models(HAPPY_SUPERVISOR, [GOOD_ANSWER])).get_graph().draw_mermaid()
    for node in ("supervisor", "researcher", "analyst", "synthesizer", "validator", "__end__"):
        assert node in mermaid
