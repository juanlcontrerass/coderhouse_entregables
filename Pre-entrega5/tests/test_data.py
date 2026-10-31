from react_agent.data import find_customer, get_order, get_orders, suggest_customers


def test_find_customer_ignores_case_and_accents():
    customer = find_customer("  lucia FERNANDEZ ")
    assert customer is not None
    assert customer.id == 102


def test_find_customer_returns_none_when_unknown():
    assert find_customer("Roberto Sánchez") is None


def test_suggest_customers_finds_misspelled_name():
    assert suggest_customers("Lusia Fernandes") == ["Lucía Fernández"]


def test_suggest_customers_is_empty_for_unrelated_name():
    assert suggest_customers("Roberto Sánchez") == []


def test_get_orders_are_sorted_oldest_first():
    dates = [order.date for order in get_orders(102)]
    assert dates == sorted(dates)
    assert len(dates) == 3


def test_get_orders_of_unknown_customer_is_empty():
    assert get_orders(999) == []


def test_order_total_is_computed_from_its_items():
    order = get_order(1003)
    assert order is not None
    assert order.total == 4500


def test_order_total_multiplies_quantity_by_price():
    order = get_order(1002)  # 2 cables de $750
    assert order is not None
    assert order.total == 1500


def test_customer_102_matches_the_example_in_the_assignment():
    assert sum(order.total for order in get_orders(102)) == 14500
