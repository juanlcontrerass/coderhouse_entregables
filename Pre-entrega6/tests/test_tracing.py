"""La búsqueda en la Vector DB emite un span RETRIEVER de OpenInference (sin Phoenix ni red)."""

import json
from collections.abc import Iterator

import pytest
from openinference.instrumentation.langchain import LangChainInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

import tools.retrieval as retrieval
from runner import run_query
from tests.test_graph import GOOD_ANSWER, HAPPY_SUPERVISOR, QUERY, _graph, _models


@pytest.fixture
def exporter(monkeypatch: pytest.MonkeyPatch) -> InMemorySpanExporter:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(retrieval, "tracer", provider.get_tracer("test"))
    return exporter


def test_search_emits_retriever_span(exporter: InMemorySpanExporter) -> None:
    result = retrieval.buscar_documentos.invoke({"consulta": "días de vacaciones", "k": 3})

    (span,) = exporter.get_finished_spans()
    attrs = span.attributes or {}
    assert span.name == "chroma.similarity_search"
    assert attrs["openinference.span.kind"] == "RETRIEVER"
    assert attrs["input.value"] == "días de vacaciones"
    assert attrs["retrieval.top_k"] == 3
    for i in range(3):
        content = attrs[f"retrieval.documents.{i}.document.content"]
        source = json.loads(attrs[f"retrieval.documents.{i}.document.metadata"])["source"]
        assert isinstance(attrs[f"retrieval.documents.{i}.document.score"], float)
        assert f"[Fuente: {source}]\n{content}" in result
    assert "retrieval.documents.3.document.content" not in attrs


@pytest.fixture
def instrumented(exporter: InMemorySpanExporter) -> Iterator[InMemorySpanExporter]:
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    LangChainInstrumentor().instrument(tracer_provider=provider)
    yield exporter
    LangChainInstrumentor().uninstrument()


async def test_retriever_span_hangs_from_the_tool_that_called_it(instrumented: InMemorySpanExporter) -> None:
    await run_query(_graph(_models(HAPPY_SUPERVISOR, [GOOD_ANSWER])), QUERY, session_id="sesion-test")

    spans = instrumented.get_finished_spans()
    by_id = {s.context.span_id: s for s in spans}
    retriever = next(s for s in spans if s.name == "chroma.similarity_search")
    ancestors = []
    parent = retriever.parent
    while parent is not None:
        ancestors.append(by_id[parent.span_id].name)
        parent = by_id[parent.span_id].parent
    assert ancestors[0] == "buscar_documentos" and ancestors[-1] == "orquestador"
    assert "researcher" in ancestors
    assert len({s.context.trace_id for s in spans}) == 1  # una sola traza por consulta
    assert (retriever.attributes or {})["session.id"] == "sesion-test"
