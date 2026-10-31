"""Herramientas del agente.

El LLM decide cuál usar leyendo solo el docstring, por eso son tan descriptivos. Los
errores se devuelven como datos (no como excepciones) para que el modelo pueda leerlos
y reintentar o pedir una aclaración.
"""

import json
from typing import Any

from langchain_core.tools import BaseTool, tool

from react_agent.data import find_customer, get_order, get_orders, suggest_customers


def _to_json(data: dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False)


@tool
async def buscar_cliente(nombre: str) -> str:
    """Busca el cliente_id de un cliente a partir de su nombre y apellido.

    Úsala primero cuando el usuario nombre a un cliente, porque las demás herramientas
    necesitan su ID numérico. No distingue mayúsculas ni acentos.

    Devuelve {"cliente_id", "nombre"} si lo encuentra. Si no, devuelve {"error",
    "sugerencias"}, donde "sugerencias" lista los nombres registrados más parecidos y
    puede estar vacía.
    """
    customer = find_customer(nombre)
    if customer is None:
        return _to_json(
            {
                "error": f"No hay ningún cliente registrado con el nombre '{nombre}'.",
                "sugerencias": suggest_customers(nombre),
            }
        )
    return _to_json({"cliente_id": customer.id, "nombre": customer.name})


@tool
async def listar_pedidos(cliente_id: int) -> str:
    """Lista los pedidos de un cliente a partir de su cliente_id numérico.

    Úsala para saber cuántos pedidos hizo un cliente, cuánto gastó en total o cuál fue
    su último pedido. Si solo tenés el nombre, primero usá buscar_cliente.

    Devuelve {"cantidad", "total", "pedidos"}. Cada pedido incluye "pedido_id", "fecha",
    "estado" y "total", y están ordenados por fecha: el último de la lista es el más
    reciente. No incluye los productos: para verlos, usá detalle_pedido. Si el cliente
    no existe, devuelve {"error"}.
    """
    orders = get_orders(cliente_id)
    if not orders:
        return _to_json({"error": f"No hay pedidos para el cliente_id {cliente_id}."})
    return _to_json(
        {
            "cantidad": len(orders),
            "total": sum(order.total for order in orders),
            "pedidos": [
                {
                    "pedido_id": order.id,
                    "fecha": order.date.isoformat(),
                    "estado": order.status,
                    "total": order.total,
                }
                for order in orders
            ],
        }
    )


@tool
async def detalle_pedido(pedido_id: int) -> str:
    """Muestra los productos de un pedido a partir de su pedido_id numérico.

    Úsala cuando pregunten qué se compró en un pedido concreto. Si no conocés el
    pedido_id, primero usá listar_pedidos.

    Devuelve {"pedido_id", "fecha", "estado", "total", "productos"}. Cada producto incluye
    "producto", "cantidad", "precio_unitario" y "subtotal". Si el pedido no existe,
    devuelve {"error"}.
    """
    order = get_order(pedido_id)
    if order is None:
        return _to_json({"error": f"No existe ningún pedido con pedido_id {pedido_id}."})
    return _to_json(
        {
            "pedido_id": order.id,
            "fecha": order.date.isoformat(),
            "estado": order.status,
            "total": order.total,
            "productos": [
                {
                    "producto": item.product,
                    "cantidad": item.quantity,
                    "precio_unitario": item.unit_price,
                    "subtotal": item.subtotal,
                }
                for item in order.items
            ],
        }
    )


TOOLS: list[BaseTool] = [buscar_cliente, listar_pedidos, detalle_pedido]
