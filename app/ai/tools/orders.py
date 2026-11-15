from langchain_core.tools import tool

ORDERS = {  # mock order DB
    "A-1001": {"plan": "Pro", "days_since_purchase": 10, "amount_usd": 20},
    "A-1002": {"plan": "Team", "days_since_purchase": 30, "amount_usd": 150},
}


def make_order_tool(orders: dict[str, dict]):
    @tool
    def get_order_status(order_id: str) -> dict:
        """Look up an order by ID. Returns plan, days since purchase and amount."""
        return orders.get(order_id.upper(), {"error": f"order {order_id} not found"})

    return get_order_status
