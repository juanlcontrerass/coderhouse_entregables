"""Base de datos simulada de clientes y pedidos.

Clientes y pedidos están separados a propósito: el usuario pregunta por nombre, pero
los pedidos se consultan por ID, así que el agente necesita varios pasos para responder.
"""

import unicodedata
from dataclasses import dataclass
from datetime import date
from difflib import get_close_matches

SUGGESTION_CUTOFF = 0.75


@dataclass(frozen=True)
class Customer:
    id: int
    name: str


@dataclass(frozen=True)
class OrderItem:
    product: str
    quantity: int
    unit_price: int

    @property
    def subtotal(self) -> int:
        return self.quantity * self.unit_price


@dataclass(frozen=True)
class Order:
    id: int
    customer_id: int
    date: date
    status: str
    items: tuple[OrderItem, ...]

    @property
    def total(self) -> int:
        return sum(item.subtotal for item in self.items)


CUSTOMERS: tuple[Customer, ...] = (
    Customer(102, "Lucía Fernández"),
    Customer(205, "Martín Rojas"),
    Customer(310, "Camila Soto"),
)

ORDERS: tuple[Order, ...] = (
    Order(1002, 102, date(2026, 6, 5), "entregado", (OrderItem("Cable USB-C", 2, 750),)),
    Order(
        1001,
        102,
        date(2026, 3, 12),
        "entregado",
        (OrderItem("Teclado mecánico", 1, 6000), OrderItem("Mouse inalámbrico", 1, 2500)),
    ),
    Order(
        1003,
        102,
        date(2026, 9, 21),
        "en camino",
        (OrderItem("Hub USB", 1, 3000), OrderItem("Soporte para notebook", 1, 1500)),
    ),
    Order(2001, 205, date(2026, 7, 2), "entregado", (OrderItem("Auriculares", 1, 3200),)),
    Order(3001, 310, date(2026, 1, 28), "entregado", (OrderItem("Monitor 27 pulgadas", 1, 18000),)),
    Order(
        3002,
        310,
        date(2026, 8, 14),
        "entregado",
        (OrderItem("Webcam", 1, 4800), OrderItem("Micrófono", 2, 2500)),
    ),
)


def _normalize(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    without_accents = "".join(c for c in decomposed if not unicodedata.combining(c))
    return " ".join(without_accents.lower().split())


_CUSTOMERS_BY_NAME = {_normalize(customer.name): customer for customer in CUSTOMERS}


def find_customer(name: str) -> Customer | None:
    return _CUSTOMERS_BY_NAME.get(_normalize(name))


def suggest_customers(name: str) -> list[str]:
    """Nombres de clientes parecidos al indicado, para corregir errores de tipeo."""
    matches = get_close_matches(_normalize(name), _CUSTOMERS_BY_NAME, cutoff=SUGGESTION_CUTOFF)
    return [_CUSTOMERS_BY_NAME[match].name for match in matches]


def get_orders(customer_id: int) -> list[Order]:
    """Pedidos de un cliente, del más antiguo al más reciente."""
    orders = (order for order in ORDERS if order.customer_id == customer_id)
    return sorted(orders, key=lambda order: order.date)


def get_order(order_id: int) -> Order | None:
    return next((order for order in ORDERS if order.id == order_id), None)
