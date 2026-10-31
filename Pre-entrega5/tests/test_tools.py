import asyncio
import json
from typing import Any

from react_agent.tools import TOOLS, buscar_cliente, detalle_pedido, listar_pedidos


def call(tool: Any, **args: Any) -> dict[str, Any]:
    return json.loads(asyncio.run(tool.ainvoke(args)))


def test_buscar_cliente_returns_id():
    assert call(buscar_cliente, nombre="lucia fernandez") == {"cliente_id": 102, "nombre": "Lucía Fernández"}


def test_buscar_cliente_misspelled_returns_error_with_suggestion():
    result = call(buscar_cliente, nombre="Lusia Fernandes")
    assert "error" in result
    assert result["sugerencias"] == ["Lucía Fernández"]


def test_buscar_cliente_unknown_returns_error_without_suggestions():
    result = call(buscar_cliente, nombre="Roberto Sánchez")
    assert "error" in result
    assert result["sugerencias"] == []


def test_listar_pedidos_returns_count_and_total():
    result = call(listar_pedidos, cliente_id=102)
    assert result["cantidad"] == 3
    assert result["total"] == 14500
    assert [p["fecha"] for p in result["pedidos"]] == sorted(p["fecha"] for p in result["pedidos"])


def test_listar_pedidos_unknown_customer_returns_error():
    assert "error" in call(listar_pedidos, cliente_id=999)


def test_detalle_pedido_returns_items():
    result = call(detalle_pedido, pedido_id=1003)
    assert result["pedido_id"] == 1003
    assert [p["subtotal"] for p in result["productos"]] == [3000, 1500]
    assert result["total"] == 4500


def test_detalle_pedido_unknown_returns_error():
    assert "error" in call(detalle_pedido, pedido_id=1)


def test_every_tool_has_a_descriptive_docstring():
    for tool in TOOLS:
        assert len(tool.description) > 100, tool.name
