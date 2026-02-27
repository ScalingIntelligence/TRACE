"""Lightweight tool executor operating on plain dict databases.

No tau2-bench dependency — works directly with synthetic dict-based DBs
from synthetic_db.py. Implements the same tool names and argument signatures
as tau2-bench so the model sees identical tool interfaces.
"""

import json
import copy
import math
from typing import Dict, Any, List, Optional
from datetime import datetime, date


def _json_response(data: Any) -> str:
    """Convert a response to JSON string, matching tau2-bench output format."""
    if isinstance(data, str):
        return data
    return json.dumps(data, default=str)


class ToolExecutor:
    """Executes tools against a plain dict database."""

    def __init__(self, domain: str, db: Dict[str, Any]):
        """Initialize with domain and dict database.

        Args:
            domain: "airline" or "retail"
            db: Dict with keys like {"users": {...}, "reservations": {...}, "flights": {...}}
                or {"users": {...}, "orders": {...}, "products": {...}}
        """
        self.domain = domain
        self._db = db
        self._tool_call_log: List[Dict[str, Any]] = []

    @property
    def db(self) -> Dict[str, Any]:
        """Return DB as plain dict (for verification)."""
        return self._db

    @property
    def tool_calls(self) -> List[Dict[str, Any]]:
        return list(self._tool_call_log)

    def execute(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Execute a tool and return result string."""
        # Normalize retail order_id: DB stores "#W..." but models often
        # strip the '#'. Auto-correct to avoid blocking training.
        if self.domain == "retail" and "order_id" in arguments:
            oid = str(arguments["order_id"])
            if oid and not oid.startswith("#"):
                arguments = {**arguments, "order_id": f"#{oid}"}

        entry = {
            "name": tool_name,
            "arguments": copy.deepcopy(arguments),
        }
        self._tool_call_log.append(entry)

        try:
            handler = self._get_handler(tool_name)
            if handler is None:
                entry["error"] = True
                return f"Error: Unknown tool '{tool_name}'"
            result = handler(**arguments)
            entry["error"] = False
            return _json_response(result)
        except Exception as e:
            entry["error"] = True
            return f"Error: {e}"

    def _get_handler(self, tool_name: str):
        """Get the handler function for a tool name."""
        # Shared tools
        shared = {
            "transfer_to_human_agents": self._transfer_to_human_agents,
            "find_user_id_by_email": self._find_user_id_by_email,
            "find_user_id_by_name_zip": self._find_user_id_by_name_zip,
        }
        if tool_name in shared:
            return shared[tool_name]

        if self.domain == "airline":
            handlers = {
                "get_user_details": self._airline_get_user_details,
                "get_reservation_details": self._airline_get_reservation_details,
                "cancel_reservation": self._airline_cancel_reservation,
                "update_reservation_flights": self._airline_update_reservation_flights,
                "update_reservation_baggages": self._airline_update_reservation_baggages,
                "update_reservation_passengers": self._airline_update_reservation_passengers,
                "book_reservation": self._airline_book_reservation,
                "send_certificate": self._airline_send_certificate,
                "search_one_way_flight": self._airline_search_flights,
                "search_round_trip_flight": self._airline_search_flights,
                "search_direct_flight": self._airline_search_flights,
                "search_onestop_flight": self._airline_search_flights,
                "list_all_airports": self._airline_list_all_airports,
                "get_flight_details": self._airline_get_flight_details,
                "calculate": self._calculate,
            }
        else:
            handlers = {
                "get_user_details": self._retail_get_user_details,
                "get_order_details": self._retail_get_order_details,
                "cancel_pending_order": self._retail_cancel_pending_order,
                "modify_pending_order_items": self._retail_modify_pending_order_items,
                "modify_pending_order_address": self._retail_modify_pending_order_address,
                "modify_pending_order_payment": self._retail_modify_pending_order_payment,
                "return_delivered_order_items": self._retail_return_delivered_order_items,
                "exchange_delivered_order_items": self._retail_exchange_delivered_order_items,
                "modify_user_address": self._retail_modify_user_address,
                "modify_user_email": self._retail_modify_user_email,
                "get_product_details": self._retail_get_product_details,
                "list_all_product_types": self._retail_list_all_product_types,
                "search_products": self._retail_search_products,
                "calculate": self._calculate,
            }
        return handlers.get(tool_name)

    # =================================================================
    # Shared tools
    # =================================================================

    def _transfer_to_human_agents(self, summary: str = "", **kwargs) -> str:
        return "Transfer successful. The user has been connected to a human agent."

    def _find_user_id_by_email(self, email: str, **kwargs) -> str:
        for uid, user in self._db.get("users", {}).items():
            if user.get("email", "").lower() == email.lower():
                return uid
        return "Error: user not found"

    def _find_user_id_by_name_zip(self, first_name: str = "", last_name: str = "",
                                   zip: str = "", **kwargs) -> str:
        for uid, user in self._db.get("users", {}).items():
            name = user.get("name", {})
            addr = user.get("address", {})
            if (name.get("first_name", "").lower() == first_name.lower() and
                name.get("last_name", "").lower() == last_name.lower() and
                addr.get("zip", "") == zip):
                return uid
        return "Error: user not found"

    def _calculate(self, expression: str = "", **kwargs):
        """Safe math expression evaluator."""
        try:
            # Only allow safe math operations
            allowed = set("0123456789+-*/.() ")
            if not all(c in allowed for c in expression):
                return f"Error: invalid expression"
            result = eval(expression, {"__builtins__": {}}, {"math": math})
            return str(result)
        except Exception as e:
            return f"Error: {e}"

    # =================================================================
    # Airline tools
    # =================================================================

    def _airline_get_user_details(self, user_id: str, **kwargs) -> Any:
        user = self._db.get("users", {}).get(user_id)
        if user is None:
            return "Error: user not found"
        return user

    def _airline_get_reservation_details(self, reservation_id: str, **kwargs) -> Any:
        res = self._db.get("reservations", {}).get(reservation_id)
        if res is None:
            return "Error: reservation not found"
        return res

    def _airline_cancel_reservation(self, reservation_id: str, **kwargs) -> Any:
        res = self._db.get("reservations", {}).get(reservation_id)
        if res is None:
            return "Error: reservation not found"
        if res.get("status") == "cancelled":
            return "Error: reservation already cancelled"
        res["status"] = "cancelled"
        return {"status": "success", "reservation_id": reservation_id, "message": "Reservation cancelled"}

    def _airline_update_reservation_flights(self, reservation_id: str, **kwargs) -> Any:
        res = self._db.get("reservations", {}).get(reservation_id)
        if res is None:
            return "Error: reservation not found"
        if res.get("status") == "cancelled":
            return "Error: reservation is cancelled"

        # Update cabin if provided
        if "cabin" in kwargs and kwargs["cabin"]:
            res["cabin"] = kwargs["cabin"]

        # Update flights if provided
        if "flights" in kwargs and kwargs["flights"]:
            flights = kwargs["flights"]
            if isinstance(flights, str):
                flights = json.loads(flights)
            res["flights"] = flights

        # Update payment if provided
        if "payment_method_id" in kwargs and kwargs["payment_method_id"]:
            res.setdefault("payment_history", []).append({
                "payment_method_id": kwargs["payment_method_id"],
                "amount": 0,  # Actual amount would be calculated
            })

        return {"status": "success", "reservation_id": reservation_id,
                "message": "Reservation updated"}

    def _airline_update_reservation_baggages(self, reservation_id: str,
                                              total_baggages: int = 0,
                                              nonfree_baggages: int = 0,
                                              payment_method_id: str = "",
                                              **kwargs) -> Any:
        res = self._db.get("reservations", {}).get(reservation_id)
        if res is None:
            return "Error: reservation not found"

        res["total_baggages"] = int(total_baggages)
        res["nonfree_baggages"] = int(nonfree_baggages)
        if payment_method_id:
            res.setdefault("payment_history", []).append({
                "payment_method_id": payment_method_id,
                "amount": int(nonfree_baggages) * 50,
            })
        return {"status": "success", "reservation_id": reservation_id,
                "total_baggages": int(total_baggages)}

    def _airline_update_reservation_passengers(self, reservation_id: str,
                                                passengers: Any = None,
                                                **kwargs) -> Any:
        res = self._db.get("reservations", {}).get(reservation_id)
        if res is None:
            return "Error: reservation not found"
        if passengers:
            if isinstance(passengers, str):
                passengers = json.loads(passengers)
            res["passengers"] = passengers
        return {"status": "success", "reservation_id": reservation_id}

    def _airline_book_reservation(self, **kwargs) -> Any:
        import random
        res_id = "".join(random.choice("ABCDEFGHJKLMNPQRSTUVWXYZ0123456789") for _ in range(6))
        return {"status": "success", "reservation_id": res_id}

    def _airline_send_certificate(self, user_id: str, amount: float = 0, **kwargs) -> Any:
        user = self._db.get("users", {}).get(user_id)
        if user is None:
            return "Error: user not found"
        return {"status": "success", "user_id": user_id, "certificate_amount": amount}

    def _airline_search_flights(self, origin: str = "", destination: str = "",
                                 date: str = "", return_date: str = "", **kwargs) -> Any:
        """Search flights in the synthetic DB."""
        results = []
        for fn, flight in self._db.get("flights", {}).items():
            if origin and flight.get("origin") != origin:
                continue
            if destination and flight.get("destination") != destination:
                continue
            if date:
                if date in flight.get("dates", {}):
                    date_info = flight["dates"][date]
                    results.append({
                        "flight_number": fn,
                        "origin": flight.get("origin"),
                        "destination": flight.get("destination"),
                        "date": date,
                        "departure_time": flight.get("scheduled_departure_time_est"),
                        "arrival_time": flight.get("scheduled_arrival_time_est"),
                        "available_seats": date_info.get("available_seats", {}),
                        "prices": date_info.get("prices", {}),
                    })
            else:
                for d, date_info in flight.get("dates", {}).items():
                    results.append({
                        "flight_number": fn,
                        "origin": flight.get("origin"),
                        "destination": flight.get("destination"),
                        "date": d,
                        "prices": date_info.get("prices", {}),
                    })
        return results if results else "No flights found matching the criteria."

    def _airline_list_all_airports(self, **kwargs) -> Any:
        from .constants import AIRPORTS
        return AIRPORTS

    def _airline_get_flight_details(self, flight_number: str, **kwargs) -> Any:
        flight = self._db.get("flights", {}).get(flight_number)
        if flight is None:
            return "Error: flight not found"
        return flight

    # =================================================================
    # Retail tools
    # =================================================================

    def _retail_get_user_details(self, user_id: str, **kwargs) -> Any:
        user = self._db.get("users", {}).get(user_id)
        if user is None:
            return "Error: user not found"
        return user

    def _retail_get_order_details(self, order_id: str, **kwargs) -> Any:
        order = self._db.get("orders", {}).get(order_id)
        if order is None:
            return "Error: order not found"
        return order

    def _retail_cancel_pending_order(self, order_id: str, reason: str = "", **kwargs) -> Any:
        order = self._db.get("orders", {}).get(order_id)
        if order is None:
            return "Error: order not found"
        if order.get("status") != "pending":
            return f"Error: order status is '{order.get('status')}', not 'pending'"
        order["status"] = "cancelled"
        order["cancel_reason"] = reason
        # Refund to original payment
        if order.get("payment_history"):
            original = order["payment_history"][0]
            order["payment_history"].append({
                "transaction_type": "refund",
                "amount": original.get("amount", 0),
                "payment_method_id": original.get("payment_method_id"),
            })
        return {"status": "success", "order_id": order_id, "message": "Order cancelled"}

    def _retail_modify_pending_order_items(self, order_id: str,
                                            item_ids: Any = None,
                                            new_item_ids: Any = None,
                                            payment_method_id: str = "",
                                            **kwargs) -> Any:
        order = self._db.get("orders", {}).get(order_id)
        if order is None:
            return "Error: order not found"
        if order.get("status") != "pending":
            return f"Error: order status is '{order.get('status')}', not 'pending'"

        if isinstance(item_ids, str):
            item_ids = json.loads(item_ids)
        if isinstance(new_item_ids, str):
            new_item_ids = json.loads(new_item_ids)

        # Replace items
        if item_ids and new_item_ids:
            for old_id, new_id in zip(item_ids, new_item_ids):
                for i, item in enumerate(order.get("items", [])):
                    if item.get("item_id") == old_id:
                        # Look up new item in products_db
                        for prod in self._db.get("products", {}).values():
                            variant = prod.get("variants", {}).get(new_id)
                            if variant:
                                order["items"][i] = {
                                    "name": prod["name"],
                                    "product_id": prod["product_id"],
                                    "item_id": new_id,
                                    "price": variant.get("price", item["price"]),
                                    "options": variant.get("options", {}),
                                }
                                break

        return {"status": "success", "order_id": order_id, "message": "Order items modified"}

    def _retail_modify_pending_order_address(self, order_id: str, **kwargs) -> Any:
        order = self._db.get("orders", {}).get(order_id)
        if order is None:
            return "Error: order not found"
        if order.get("status") != "pending":
            return f"Error: order status is '{order.get('status')}', not 'pending'"

        new_addr = {}
        for key in ["address1", "address2", "city", "state", "country", "zip"]:
            if key in kwargs:
                new_addr[key] = kwargs[key]
        if new_addr:
            order["address"] = {**order.get("address", {}), **new_addr}
        return {"status": "success", "order_id": order_id, "message": "Address updated"}

    def _retail_modify_pending_order_payment(self, order_id: str,
                                              payment_method_id: str = "",
                                              **kwargs) -> Any:
        order = self._db.get("orders", {}).get(order_id)
        if order is None:
            return "Error: order not found"
        if order.get("status") != "pending":
            return f"Error: order status is '{order.get('status')}', not 'pending'"
        if payment_method_id:
            order["payment_history"].append({
                "transaction_type": "payment_update",
                "payment_method_id": payment_method_id,
            })
        return {"status": "success", "order_id": order_id, "message": "Payment updated"}

    def _retail_return_delivered_order_items(self, order_id: str,
                                              item_ids: Any = None,
                                              payment_method_id: str = "",
                                              **kwargs) -> Any:
        order = self._db.get("orders", {}).get(order_id)
        if order is None:
            return "Error: order not found"
        if order.get("status") != "delivered":
            return f"Error: order status is '{order.get('status')}', not 'delivered'"

        if isinstance(item_ids, str):
            item_ids = json.loads(item_ids)

        order["return_items"] = item_ids or [i["item_id"] for i in order.get("items", [])]
        order["return_payment_method_id"] = payment_method_id
        order["status"] = "return requested"

        # Calculate refund
        refund = sum(
            item["price"] for item in order.get("items", [])
            if item["item_id"] in (order["return_items"] or [])
        )
        order["payment_history"].append({
            "transaction_type": "refund",
            "amount": refund,
            "payment_method_id": payment_method_id,
        })
        return {"status": "success", "order_id": order_id,
                "message": "Return initiated", "refund_amount": refund}

    def _retail_exchange_delivered_order_items(self, order_id: str,
                                                item_ids: Any = None,
                                                new_item_ids: Any = None,
                                                payment_method_id: str = "",
                                                **kwargs) -> Any:
        order = self._db.get("orders", {}).get(order_id)
        if order is None:
            return "Error: order not found"
        if order.get("status") != "delivered":
            return f"Error: order status is '{order.get('status')}', not 'delivered'"

        if isinstance(item_ids, str):
            item_ids = json.loads(item_ids)
        if isinstance(new_item_ids, str):
            new_item_ids = json.loads(new_item_ids)

        order["exchange_items"] = item_ids
        order["exchange_new_items"] = new_item_ids
        order["exchange_payment_method_id"] = payment_method_id
        order["status"] = "exchange requested"
        return {"status": "success", "order_id": order_id, "message": "Exchange initiated"}

    def _retail_modify_user_address(self, user_id: str, **kwargs) -> Any:
        user = self._db.get("users", {}).get(user_id)
        if user is None:
            return "Error: user not found"
        new_addr = {}
        for key in ["address1", "address2", "city", "state", "country", "zip"]:
            if key in kwargs:
                new_addr[key] = kwargs[key]
        if new_addr:
            user["address"] = {**user.get("address", {}), **new_addr}
        return {"status": "success", "user_id": user_id, "message": "Address updated"}

    def _retail_modify_user_email(self, user_id: str, new_email: str = "", **kwargs) -> Any:
        user = self._db.get("users", {}).get(user_id)
        if user is None:
            return "Error: user not found"
        user["email"] = new_email
        return {"status": "success", "user_id": user_id, "message": "Email updated"}

    def _retail_get_product_details(self, product_id: str, **kwargs) -> Any:
        product = self._db.get("products", {}).get(product_id)
        if product is None:
            return "Error: product not found"
        return product

    def _retail_list_all_product_types(self, **kwargs) -> Any:
        types = set()
        for prod in self._db.get("products", {}).values():
            types.add(prod.get("name", "unknown"))
        return sorted(types)

    def _retail_search_products(self, name: str = "", **kwargs) -> Any:
        results = []
        for pid, prod in self._db.get("products", {}).items():
            if name and name.lower() not in prod.get("name", "").lower():
                continue
            results.append({
                "product_id": pid,
                "name": prod.get("name"),
                "variants": {
                    vid: {
                        "item_id": v.get("item_id"),
                        "options": v.get("options"),
                        "available": v.get("available"),
                        "price": v.get("price"),
                    }
                    for vid, v in prod.get("variants", {}).items()
                },
            })
        return results if results else "No products found."
