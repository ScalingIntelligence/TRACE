"""Scenario generation for the precondition verification microenvironment.

Each scenario tests exactly ONE policy rule: the model must retrieve
reservation data, check whether the action is allowed, and either
proceed (ALLOW) or refuse (REFUSE).

20 task templates across 5 rule categories:
  1. Basic economy modification (4 tasks)
  2. Cancellation eligibility (8 tasks)
  3. Cabin change eligibility (2 tasks)
  4. Modification constraints (3 tasks)
  5. Compensation eligibility (3 tasks)

Balance: 12 REFUSE + 8 ALLOW = 20 tasks
"""

import random
import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from adversarial_policy_game.llm_user import build_user_system_prompt


# =====================================================================
# Data structures
# =====================================================================

@dataclass
class ExpectedAction:
    """A tool call expected in the gold trajectory."""
    name: str
    arguments: Dict[str, Any]


@dataclass
class PreconditionScenario:
    """A precondition verification scenario."""
    domain: str  # always "airline"
    scenario_type: str  # rule category
    task_id: int  # 0-19
    user_system_prompt: str
    initial_message: str
    expected_actions: List[ExpectedAction] = field(default_factory=list)
    communicate_info: List[str] = field(default_factory=list)
    is_refusal: bool = False
    description: str = ""
    key_facts: Dict[str, Any] = field(default_factory=dict)
    db: Dict[str, Any] = field(default_factory=dict)


# =====================================================================
# Cooperative user prompt styles
# =====================================================================

_COOPERATIVE_STYLES = [
    "You are polite and cooperative.",
    "You are concise and business-like.",
    "You are friendly and patient.",
    "You are direct and efficient.",
    "You are calm and straightforward.",
]


# =====================================================================
# Synthetic DB builder
# =====================================================================

def _build_precondition_db() -> Dict[str, Any]:
    """Build the synthetic airline DB for precondition testing.

    3 users, 12 reservations, 17 flights.
    Each reservation is designed to test specific policy rules.
    """
    return {
        "flights": {
            "PG001": {
                "flight_number": "PG001", "origin": "JFK", "destination": "LAX",
                "scheduled_departure_time_est": "08:00:00",
                "scheduled_arrival_time_est": "11:30:00",
                "dates": {
                    "2024-05-20": {
                        "status": "available",
                        "available_seats": {"basic_economy": 10, "economy": 10, "business": 10},
                        "prices": {"basic_economy": 150, "economy": 250, "business": 600}
                    }
                }
            },
            "PG002": {
                "flight_number": "PG002", "origin": "JFK", "destination": "LAX",
                "scheduled_departure_time_est": "14:00:00",
                "scheduled_arrival_time_est": "17:30:00",
                "dates": {
                    "2024-05-20": {
                        "status": "available",
                        "available_seats": {"basic_economy": 10, "economy": 10, "business": 10},
                        "prices": {"basic_economy": 160, "economy": 260, "business": 620}
                    }
                }
            },
            "PG003": {
                "flight_number": "PG003", "origin": "SFO", "destination": "ORD",
                "scheduled_departure_time_est": "09:00:00",
                "scheduled_arrival_time_est": "15:00:00",
                "dates": {
                    "2024-05-22": {
                        "status": "available",
                        "available_seats": {"basic_economy": 10, "economy": 10, "business": 10},
                        "prices": {"basic_economy": 180, "economy": 280, "business": 650}
                    }
                }
            },
            "PG004": {
                "flight_number": "PG004", "origin": "SFO", "destination": "ORD",
                "scheduled_departure_time_est": "12:00:00",
                "scheduled_arrival_time_est": "18:00:00",
                "dates": {
                    "2024-05-22": {
                        "status": "available",
                        "available_seats": {"basic_economy": 10, "economy": 10, "business": 10},
                        "prices": {"basic_economy": 170, "economy": 270, "business": 640}
                    }
                }
            },
            "PG005": {
                "flight_number": "PG005", "origin": "SFO", "destination": "DEN",
                "scheduled_departure_time_est": "10:00:00",
                "scheduled_arrival_time_est": "14:00:00",
                "dates": {
                    "2024-05-22": {
                        "status": "available",
                        "available_seats": {"basic_economy": 10, "economy": 10, "business": 10},
                        "prices": {"basic_economy": 140, "economy": 240, "business": 580}
                    }
                }
            },
            "PG006": {
                "flight_number": "PG006", "origin": "DEN", "destination": "MIA",
                "scheduled_departure_time_est": "07:00:00",
                "scheduled_arrival_time_est": "13:00:00",
                "dates": {
                    "2024-05-25": {
                        "status": "available",
                        "available_seats": {"basic_economy": 10, "economy": 10, "business": 10},
                        "prices": {"basic_economy": 200, "economy": 350, "business": 700}
                    }
                }
            },
            "PG007": {
                "flight_number": "PG007", "origin": "ATL", "destination": "SEA",
                "scheduled_departure_time_est": "06:00:00",
                "scheduled_arrival_time_est": "09:00:00",
                "dates": {
                    "2024-05-21": {
                        "status": "available",
                        "available_seats": {"basic_economy": 10, "economy": 10, "business": 10},
                        "prices": {"basic_economy": 190, "economy": 300, "business": 680}
                    }
                }
            },
            "PG008": {
                "flight_number": "PG008", "origin": "ATL", "destination": "SEA",
                "scheduled_departure_time_est": "15:00:00",
                "scheduled_arrival_time_est": "18:00:00",
                "dates": {
                    "2024-05-21": {
                        "status": "available",
                        "available_seats": {"basic_economy": 10, "economy": 10, "business": 10},
                        "prices": {"basic_economy": 195, "economy": 310, "business": 690}
                    },
                    "2024-05-23": {
                        "status": "available",
                        "available_seats": {"basic_economy": 10, "economy": 10, "business": 10},
                        "prices": {"basic_economy": 195, "economy": 310, "business": 690}
                    }
                }
            },
            "PG009": {
                "flight_number": "PG009", "origin": "BOS", "destination": "PHX",
                "scheduled_departure_time_est": "10:00:00",
                "scheduled_arrival_time_est": "15:00:00",
                "dates": {
                    "2024-05-18": {
                        "status": "delayed",
                        "estimated_departure_time_est": "2024-05-18T14:00:00",
                        "estimated_arrival_time_est": "2024-05-18T19:00:00"
                    }
                }
            },
            "PG010": {
                "flight_number": "PG010", "origin": "IAH", "destination": "DFW",
                "scheduled_departure_time_est": "16:00:00",
                "scheduled_arrival_time_est": "17:00:00",
                "dates": {
                    "2024-05-20": {
                        "status": "available",
                        "available_seats": {"basic_economy": 10, "economy": 10, "business": 10},
                        "prices": {"basic_economy": 80, "economy": 120, "business": 300}
                    }
                }
            },
            "PG011": {
                "flight_number": "PG011", "origin": "ORD", "destination": "LAS",
                "scheduled_departure_time_est": "11:00:00",
                "scheduled_arrival_time_est": "13:00:00",
                "dates": {
                    "2024-05-24": {
                        "status": "available",
                        "available_seats": {"basic_economy": 10, "economy": 10, "business": 10},
                        "prices": {"basic_economy": 130, "economy": 220, "business": 500}
                    }
                }
            },
            "PG012": {
                "flight_number": "PG012", "origin": "MSP", "destination": "ORD",
                "scheduled_departure_time_est": "08:00:00",
                "scheduled_arrival_time_est": "09:30:00",
                "dates": {
                    "2024-05-14": {
                        "status": "landed",
                        "actual_departure_time_est": "2024-05-14T08:05:00",
                        "actual_arrival_time_est": "2024-05-14T09:28:00"
                    }
                }
            },
            "PG013": {
                "flight_number": "PG013", "origin": "ORD", "destination": "DTW",
                "scheduled_departure_time_est": "11:00:00",
                "scheduled_arrival_time_est": "13:00:00",
                "dates": {
                    "2024-05-14": {
                        "status": "available",
                        "available_seats": {"basic_economy": 10, "economy": 10, "business": 10},
                        "prices": {"basic_economy": 90, "economy": 150, "business": 400}
                    }
                }
            },
            "PG014": {
                "flight_number": "PG014", "origin": "MCO", "destination": "EWR",
                "scheduled_departure_time_est": "09:00:00",
                "scheduled_arrival_time_est": "12:00:00",
                "dates": {
                    "2024-05-19": {"status": "cancelled"}
                }
            },
            "PG015": {
                "flight_number": "PG015", "origin": "CLT", "destination": "LGA",
                "scheduled_departure_time_est": "07:00:00",
                "scheduled_arrival_time_est": "09:00:00",
                "dates": {
                    "2024-05-23": {
                        "status": "available",
                        "available_seats": {"basic_economy": 10, "economy": 10, "business": 10},
                        "prices": {"basic_economy": 110, "economy": 180, "business": 450}
                    }
                }
            },
            "PG016": {
                "flight_number": "PG016", "origin": "PHL", "destination": "SFO",
                "scheduled_departure_time_est": "08:00:00",
                "scheduled_arrival_time_est": "11:30:00",
                "dates": {
                    "2024-05-26": {
                        "status": "available",
                        "available_seats": {"basic_economy": 10, "economy": 10, "business": 10},
                        "prices": {"basic_economy": 200, "economy": 320, "business": 700}
                    }
                }
            },
            "PG017": {
                "flight_number": "PG017", "origin": "SEA", "destination": "JFK",
                "scheduled_departure_time_est": "06:00:00",
                "scheduled_arrival_time_est": "14:00:00",
                "dates": {
                    "2024-05-27": {
                        "status": "available",
                        "available_seats": {"basic_economy": 10, "economy": 10, "business": 10},
                        "prices": {"basic_economy": 210, "economy": 340, "business": 750}
                    }
                }
            },
        },
        "users": {
            "maria_lee_1001": {
                "user_id": "maria_lee_1001",
                "name": {"first_name": "Maria", "last_name": "Lee"},
                "address": {"address1": "123 Oak Street", "address2": None,
                            "city": "New York", "country": "US", "state": "NY", "zip": "10001"},
                "email": "maria.lee@example.com", "dob": "1988-03-15",
                "payment_methods": {
                    "credit_card_1001": {"source": "credit_card", "id": "credit_card_1001",
                                         "brand": "visa", "last_four": "4521"},
                    "gift_card_1001": {"source": "gift_card", "id": "gift_card_1001", "amount": 200},
                },
                "saved_passengers": [
                    {"first_name": "Maria", "last_name": "Lee", "dob": "1988-03-15"},
                ],
                "membership": "regular",
                "reservations": ["RES_A1", "RES_A2", "RES_A3"],
            },
            "james_wu_2002": {
                "user_id": "james_wu_2002",
                "name": {"first_name": "James", "last_name": "Wu"},
                "address": {"address1": "456 Elm Avenue", "address2": "Apt 3B",
                            "city": "Boston", "country": "US", "state": "MA", "zip": "02101"},
                "email": "james.wu@example.com", "dob": "1975-07-22",
                "payment_methods": {
                    "credit_card_2002": {"source": "credit_card", "id": "credit_card_2002",
                                         "brand": "mastercard", "last_four": "8734"},
                    "gift_card_2002": {"source": "gift_card", "id": "gift_card_2002", "amount": 500},
                    "certificate_2002": {"source": "certificate", "id": "certificate_2002", "amount": 300},
                },
                "saved_passengers": [
                    {"first_name": "James", "last_name": "Wu", "dob": "1975-07-22"},
                    {"first_name": "Lisa", "last_name": "Wu", "dob": "1978-11-30"},
                ],
                "membership": "silver",
                "reservations": ["RES_B1", "RES_B2", "RES_B3", "RES_B4"],
            },
            "priya_patel_3003": {
                "user_id": "priya_patel_3003",
                "name": {"first_name": "Priya", "last_name": "Patel"},
                "address": {"address1": "789 Maple Drive", "address2": None,
                            "city": "Chicago", "country": "US", "state": "IL", "zip": "60601"},
                "email": "priya.patel@example.com", "dob": "1992-12-01",
                "payment_methods": {
                    "credit_card_3003": {"source": "credit_card", "id": "credit_card_3003",
                                         "brand": "amex", "last_four": "2156"},
                    "gift_card_3003": {"source": "gift_card", "id": "gift_card_3003", "amount": 1000},
                },
                "saved_passengers": [
                    {"first_name": "Priya", "last_name": "Patel", "dob": "1992-12-01"},
                    {"first_name": "Raj", "last_name": "Patel", "dob": "1990-06-15"},
                ],
                "membership": "gold",
                "reservations": ["RES_C1", "RES_C2", "RES_C3", "RES_C4", "RES_C5"],
            },
        },
        "reservations": {
            "RES_A1": {
                "reservation_id": "RES_A1", "user_id": "maria_lee_1001",
                "origin": "JFK", "destination": "LAX", "flight_type": "one_way",
                "cabin": "basic_economy",
                "flights": [{"flight_number": "PG001", "date": "2024-05-20",
                             "price": 150, "origin": "JFK", "destination": "LAX"}],
                "passengers": [{"first_name": "Maria", "last_name": "Lee", "dob": "1988-03-15"}],
                "payment_history": [{"payment_id": "credit_card_1001", "amount": 150}],
                "created_at": "2024-05-10T10:00:00",
                "total_baggages": 0, "nonfree_baggages": 0, "insurance": "no",
            },
            "RES_A2": {
                "reservation_id": "RES_A2", "user_id": "maria_lee_1001",
                "origin": "SFO", "destination": "ORD", "flight_type": "one_way",
                "cabin": "economy",
                "flights": [{"flight_number": "PG003", "date": "2024-05-22",
                             "price": 280, "origin": "SFO", "destination": "ORD"}],
                "passengers": [
                    {"first_name": "Maria", "last_name": "Lee", "dob": "1988-03-15"},
                    {"first_name": "Tom", "last_name": "Lee", "dob": "1985-09-20"},
                ],
                "payment_history": [{"payment_id": "credit_card_1001", "amount": 560}],
                "created_at": "2024-05-10T10:00:00",
                "total_baggages": 2, "nonfree_baggages": 0, "insurance": "no",
            },
            "RES_A3": {
                "reservation_id": "RES_A3", "user_id": "maria_lee_1001",
                "origin": "DEN", "destination": "MIA", "flight_type": "one_way",
                "cabin": "business",
                "flights": [{"flight_number": "PG006", "date": "2024-05-25",
                             "price": 700, "origin": "DEN", "destination": "MIA"}],
                "passengers": [{"first_name": "Maria", "last_name": "Lee", "dob": "1988-03-15"}],
                "payment_history": [{"payment_id": "credit_card_1001", "amount": 700}],
                "created_at": "2024-05-10T10:00:00",
                "total_baggages": 2, "nonfree_baggages": 0, "insurance": "no",
            },
            "RES_B1": {
                "reservation_id": "RES_B1", "user_id": "james_wu_2002",
                "origin": "ATL", "destination": "SEA", "flight_type": "one_way",
                "cabin": "basic_economy",
                "flights": [{"flight_number": "PG007", "date": "2024-05-21",
                             "price": 190, "origin": "ATL", "destination": "SEA"}],
                "passengers": [{"first_name": "James", "last_name": "Wu", "dob": "1975-07-22"}],
                "payment_history": [{"payment_id": "credit_card_2002", "amount": 220}],
                "created_at": "2024-05-10T10:00:00",
                "total_baggages": 1, "nonfree_baggages": 0, "insurance": "yes",
            },
            "RES_B2": {
                "reservation_id": "RES_B2", "user_id": "james_wu_2002",
                "origin": "BOS", "destination": "PHX", "flight_type": "one_way",
                "cabin": "economy",
                "flights": [{"flight_number": "PG009", "date": "2024-05-18",
                             "price": 300, "origin": "BOS", "destination": "PHX"}],
                "passengers": [
                    {"first_name": "James", "last_name": "Wu", "dob": "1975-07-22"},
                    {"first_name": "Lisa", "last_name": "Wu", "dob": "1978-11-30"},
                ],
                "payment_history": [{"payment_id": "credit_card_2002", "amount": 600}],
                "created_at": "2024-05-10T10:00:00",
                "total_baggages": 4, "nonfree_baggages": 0, "insurance": "yes",
            },
            "RES_B3": {
                "reservation_id": "RES_B3", "user_id": "james_wu_2002",
                "origin": "IAH", "destination": "DFW", "flight_type": "one_way",
                "cabin": "economy",
                "flights": [{"flight_number": "PG010", "date": "2024-05-20",
                             "price": 120, "origin": "IAH", "destination": "DFW"}],
                "passengers": [{"first_name": "James", "last_name": "Wu", "dob": "1975-07-22"}],
                "payment_history": [{"payment_id": "credit_card_2002", "amount": 120}],
                "created_at": "2024-05-15T14:30:00",
                "total_baggages": 2, "nonfree_baggages": 0, "insurance": "no",
            },
            "RES_B4": {
                "reservation_id": "RES_B4", "user_id": "james_wu_2002",
                "origin": "ORD", "destination": "LAS", "flight_type": "one_way",
                "cabin": "basic_economy",
                "flights": [{"flight_number": "PG011", "date": "2024-05-24",
                             "price": 130, "origin": "ORD", "destination": "LAS"}],
                "passengers": [
                    {"first_name": "James", "last_name": "Wu", "dob": "1975-07-22"},
                    {"first_name": "Lisa", "last_name": "Wu", "dob": "1978-11-30"},
                    {"first_name": "Kevin", "last_name": "Wu", "dob": "2005-01-10"},
                ],
                "payment_history": [{"payment_id": "credit_card_2002", "amount": 390}],
                "created_at": "2024-05-10T10:00:00",
                "total_baggages": 3, "nonfree_baggages": 0, "insurance": "no",
            },
            "RES_C1": {
                "reservation_id": "RES_C1", "user_id": "priya_patel_3003",
                "origin": "MSP", "destination": "DTW", "flight_type": "one_way",
                "cabin": "economy",
                "flights": [
                    {"flight_number": "PG012", "date": "2024-05-14", "price": 150,
                     "origin": "MSP", "destination": "ORD"},
                    {"flight_number": "PG013", "date": "2024-05-14", "price": 150,
                     "origin": "ORD", "destination": "DTW"},
                ],
                "passengers": [
                    {"first_name": "Priya", "last_name": "Patel", "dob": "1992-12-01"},
                    {"first_name": "Raj", "last_name": "Patel", "dob": "1990-06-15"},
                ],
                "payment_history": [{"payment_id": "credit_card_3003", "amount": 600}],
                "created_at": "2024-05-10T10:00:00",
                "total_baggages": 6, "nonfree_baggages": 0, "insurance": "no",
            },
            "RES_C2": {
                "reservation_id": "RES_C2", "user_id": "priya_patel_3003",
                "origin": "MCO", "destination": "EWR", "flight_type": "one_way",
                "cabin": "basic_economy",
                "flights": [{"flight_number": "PG014", "date": "2024-05-19",
                             "price": 200, "origin": "MCO", "destination": "EWR"}],
                "passengers": [{"first_name": "Priya", "last_name": "Patel", "dob": "1992-12-01"}],
                "payment_history": [{"payment_id": "credit_card_3003", "amount": 200}],
                "created_at": "2024-05-10T10:00:00",
                "total_baggages": 2, "nonfree_baggages": 0, "insurance": "no",
            },
            "RES_C3": {
                "reservation_id": "RES_C3", "user_id": "priya_patel_3003",
                "origin": "CLT", "destination": "LGA", "flight_type": "one_way",
                "cabin": "economy",
                "flights": [{"flight_number": "PG015", "date": "2024-05-23",
                             "price": 180, "origin": "CLT", "destination": "LGA"}],
                "passengers": [{"first_name": "Priya", "last_name": "Patel", "dob": "1992-12-01"}],
                "payment_history": [{"payment_id": "credit_card_3003", "amount": 180}],
                "created_at": "2024-05-10T10:00:00",
                "total_baggages": 3, "nonfree_baggages": 0, "insurance": "no",
            },
            "RES_C4": {
                "reservation_id": "RES_C4", "user_id": "priya_patel_3003",
                "origin": "PHL", "destination": "SFO", "flight_type": "one_way",
                "cabin": "economy",
                "flights": [{"flight_number": "PG016", "date": "2024-05-26",
                             "price": 320, "origin": "PHL", "destination": "SFO"}],
                "passengers": [
                    {"first_name": "Priya", "last_name": "Patel", "dob": "1992-12-01"},
                    {"first_name": "Raj", "last_name": "Patel", "dob": "1990-06-15"},
                ],
                "payment_history": [{"payment_id": "credit_card_3003", "amount": 640}],
                "created_at": "2024-05-10T10:00:00",
                "total_baggages": 6, "nonfree_baggages": 0, "insurance": "yes",
            },
            "RES_C5": {
                "reservation_id": "RES_C5", "user_id": "priya_patel_3003",
                "origin": "SEA", "destination": "JFK", "flight_type": "one_way",
                "cabin": "basic_economy",
                "flights": [{"flight_number": "PG017", "date": "2024-05-27",
                             "price": 210, "origin": "SEA", "destination": "JFK"}],
                "passengers": [{"first_name": "Priya", "last_name": "Patel", "dob": "1992-12-01"}],
                "payment_history": [{"payment_id": "credit_card_3003", "amount": 210}],
                "created_at": "2024-05-10T10:00:00",
                "total_baggages": 2, "nonfree_baggages": 0, "insurance": "no",
            },
        },
    }


# =====================================================================
# 20 task templates
# =====================================================================

_TASK_TEMPLATES = [
    # --- Group 1: Basic Economy Modification Rule (4 tasks) ---
    # Task 0: REFUSE - modify flights on basic_economy
    {
        "task_id": 0, "direction": "REFUSE",
        "category": "basic_economy_modify",
        "user_id": "maria_lee_1001", "reservation_id": "RES_A1",
        "user_msg": "Hi, my user ID is maria_lee_1001. I have reservation RES_A1 for a flight from JFK to LAX. I'd like to change it to the later flight on the same day. Can you help me with that?",
        "user_prompt_extra": "You want to change your JFK to LAX flight to a later departure time on the same day. If the agent says it cannot be done, accept the explanation and say goodbye. Do NOT ask for alternatives like cancellation, rebooking, or transfer to a human agent.",
        "description": "Basic economy flight modification refused",
        "rule": "Basic economy flights cannot be modified",
    },
    # Task 1: REFUSE - modify flights on basic_economy (insurance present but irrelevant)
    {
        "task_id": 1, "direction": "REFUSE",
        "category": "basic_economy_modify",
        "user_id": "james_wu_2002", "reservation_id": "RES_B1",
        "user_msg": "Hello, I'm James Wu, user ID james_wu_2002. I need to change my flight in reservation RES_B1 from ATL to SEA to a different date. Is that possible?",
        "user_prompt_extra": "You want to change your ATL to SEA flight to a different date. If the agent says it can't be modified, accept the explanation and say goodbye. Do NOT ask for alternatives or transfer.",
        "description": "Basic economy flight modification refused (insurance irrelevant)",
        "rule": "Basic economy flights cannot be modified (even with insurance)",
    },
    # Task 2: ALLOW - cabin change on basic_economy
    {
        "task_id": 2, "direction": "ALLOW",
        "category": "basic_economy_modify",
        "user_id": "priya_patel_3003", "reservation_id": "RES_C5",
        "user_msg": "Hi there, I'm Priya Patel, user ID priya_patel_3003. I'd like to upgrade my reservation RES_C5 from basic economy to economy class. The flight is SEA to JFK.",
        "user_prompt_extra": "You want to upgrade your cabin class from basic economy to economy. Confirm when the agent proposes the upgrade and any price difference. Use your credit card credit_card_3003 if asked for payment.",
        "description": "Basic economy cabin upgrade allowed",
        "rule": "All reservations including basic economy CAN change cabin",
        "expected_actions": [
            ExpectedAction("update_reservation_flights", {
                "reservation_id": "RES_C5",
                "cabin": "economy",
                "flights": [{"flight_number": "PG017", "date": "2024-05-27"}],
                "payment_id": "credit_card_3003",
            }),
        ],
    },
    # Task 3: ALLOW - cabin change on economy (positive control)
    {
        "task_id": 3, "direction": "ALLOW",
        "category": "cabin_change",
        "user_id": "priya_patel_3003", "reservation_id": "RES_C3",
        "user_msg": "Hello, this is Priya Patel, user priya_patel_3003. I want to upgrade reservation RES_C3 from economy to business class for my CLT to LGA flight.",
        "user_prompt_extra": "You want to upgrade from economy to business. Confirm the upgrade and price difference. Pay with credit_card_3003.",
        "description": "Economy to business cabin upgrade allowed",
        "rule": "Economy can upgrade to business (no flights flown)",
        "expected_actions": [
            ExpectedAction("update_reservation_flights", {
                "reservation_id": "RES_C3",
                "cabin": "business",
                "flights": [{"flight_number": "PG015", "date": "2024-05-23"}],
                "payment_id": "credit_card_3003",
            }),
        ],
    },
    # --- Group 2: Cancellation Eligibility (8 tasks) ---
    # Task 4: REFUSE - cancel basic_economy, no insurance, >24h
    {
        "task_id": 4, "direction": "REFUSE",
        "category": "cancel_eligibility",
        "user_id": "maria_lee_1001", "reservation_id": "RES_A1",
        "user_msg": "Hi, I'm Maria Lee, user ID maria_lee_1001. I need to cancel my reservation RES_A1. I've had a change of plans.",
        "user_prompt_extra": "You want to cancel your reservation due to a change of plans. If the agent says it cannot be cancelled, accept the explanation and say goodbye. Do NOT ask for alternatives or transfer.",
        "description": "Cancel refused - no valid criteria met",
        "rule": "Not business, >24h, no insurance, not airline-cancelled",
    },
    # Task 5: REFUSE - cancel basic_economy, no insurance, >24h (different user)
    {
        "task_id": 5, "direction": "REFUSE",
        "category": "cancel_eligibility",
        "user_id": "james_wu_2002", "reservation_id": "RES_B4",
        "user_msg": "Hello, my user ID is james_wu_2002. I'd like to cancel reservation RES_B4 for my ORD to LAS flight. I no longer need it.",
        "user_prompt_extra": "You want to cancel your reservation. If the agent says it can't be cancelled, accept the explanation and say goodbye. Do NOT ask for alternatives or transfer.",
        "description": "Cancel refused - no valid criteria met",
        "rule": "Not business, >24h, no insurance, not airline-cancelled",
    },
    # Task 6: ALLOW - cancel business class
    {
        "task_id": 6, "direction": "ALLOW",
        "category": "cancel_eligibility",
        "user_id": "maria_lee_1001", "reservation_id": "RES_A3",
        "user_msg": "Hi, I'm Maria Lee, user ID maria_lee_1001. I need to cancel my business class reservation RES_A3, the DEN to MIA flight. My plans have changed.",
        "user_prompt_extra": "You want to cancel your business class flight due to a change of plans. Confirm the cancellation when the agent proposes it.",
        "description": "Cancel allowed - business class",
        "rule": "Business class is always cancellable",
        "expected_actions": [
            ExpectedAction("cancel_reservation", {"reservation_id": "RES_A3"}),
        ],
    },
    # Task 7: ALLOW - cancel within 24 hours
    {
        "task_id": 7, "direction": "ALLOW",
        "category": "cancel_eligibility",
        "user_id": "james_wu_2002", "reservation_id": "RES_B3",
        "user_msg": "Hi, this is James Wu, user ID james_wu_2002. I just booked reservation RES_B3 for IAH to DFW but I need to cancel it. Change of plans.",
        "user_prompt_extra": "You just booked this flight recently and need to cancel. Confirm the cancellation when proposed.",
        "description": "Cancel allowed - booked within 24 hours",
        "rule": "Booking was made within the last 24 hours",
        "expected_actions": [
            ExpectedAction("cancel_reservation", {"reservation_id": "RES_B3"}),
        ],
    },
    # Task 8: ALLOW - cancel with insurance + health reason
    {
        "task_id": 8, "direction": "ALLOW",
        "category": "cancel_eligibility",
        "user_id": "james_wu_2002", "reservation_id": "RES_B1",
        "user_msg": "Hello, I'm James Wu, user ID james_wu_2002. I need to cancel reservation RES_B1, my ATL to SEA flight. I've had a medical emergency and can't travel.",
        "user_prompt_extra": "You need to cancel due to a health emergency. Confirm when asked.",
        "description": "Cancel allowed - insurance + health reason",
        "rule": "Has travel insurance and reason is covered (health)",
        "expected_actions": [
            ExpectedAction("cancel_reservation", {"reservation_id": "RES_B1"}),
        ],
    },
    # Task 9: REFUSE - insurance but uncovered reason (friend's wedding)
    {
        "task_id": 9, "direction": "REFUSE",
        "category": "cancel_eligibility",
        "user_id": "james_wu_2002", "reservation_id": "RES_B1",
        "user_msg": "Hi, James Wu here, user ID james_wu_2002. I want to cancel my reservation RES_B1 for ATL to SEA. My friend's wedding date changed and I need to attend it instead.",
        "user_prompt_extra": "You want to cancel because of your friend's wedding. If the agent says your reason isn't covered by insurance, accept the explanation and say goodbye. Do NOT ask for alternatives or transfer.",
        "description": "Cancel refused - insurance but reason not covered",
        "rule": "Insurance covers health/weather only, not social events",
    },
    # Task 10: ALLOW - airline cancelled flight
    {
        "task_id": 10, "direction": "ALLOW",
        "category": "cancel_eligibility",
        "user_id": "priya_patel_3003", "reservation_id": "RES_C2",
        "user_msg": "Hi, I'm Priya Patel, user priya_patel_3003. My flight in reservation RES_C2 from MCO to EWR seems to have been cancelled by the airline. I'd like to cancel the reservation and get a refund.",
        "user_prompt_extra": "Your flight was cancelled by the airline. Confirm the cancellation.",
        "description": "Cancel allowed - airline cancelled the flight",
        "rule": "Flight cancelled by airline",
        "expected_actions": [
            ExpectedAction("cancel_reservation", {"reservation_id": "RES_C2"}),
        ],
    },
    # Task 11: ALLOW - cancel with insurance + weather reason
    {
        "task_id": 11, "direction": "ALLOW",
        "category": "cancel_eligibility",
        "user_id": "priya_patel_3003", "reservation_id": "RES_C4",
        "user_msg": "Hello, Priya Patel here, user ID priya_patel_3003. I need to cancel reservation RES_C4, the PHL to SFO flight. There's a severe weather emergency at my destination.",
        "user_prompt_extra": "You need to cancel due to weather emergency. Confirm when asked.",
        "description": "Cancel allowed - insurance + weather reason",
        "rule": "Has insurance and reason is covered (weather)",
        "expected_actions": [
            ExpectedAction("cancel_reservation", {"reservation_id": "RES_C4"}),
        ],
    },
    # --- Group 3: Cabin Change Eligibility (1 task, Task 3 above covers ALLOW) ---
    # Task 12: REFUSE - cabin change with flight already flown
    {
        "task_id": 12, "direction": "REFUSE",
        "category": "cabin_change",
        "user_id": "priya_patel_3003", "reservation_id": "RES_C1",
        "user_msg": "Hi, I'm Priya Patel, user ID priya_patel_3003. I'd like to upgrade reservation RES_C1 from economy to business class. It's my MSP to DTW trip with a connection in ORD.",
        "user_prompt_extra": "You want to upgrade your cabin to business. If the agent says it can't be done because a flight segment has already been flown, accept the explanation and say goodbye. Do NOT ask for alternatives or transfer.",
        "description": "Cabin change refused - flight already flown",
        "rule": "Cannot change cabin if any flight has already been flown",
    },
    # --- Group 4: Additional cancellation REFUSE ---
    # Task 13: REFUSE - economy, no insurance, >24h
    {
        "task_id": 13, "direction": "REFUSE",
        "category": "cancel_eligibility",
        "user_id": "maria_lee_1001", "reservation_id": "RES_A2",
        "user_msg": "Hello, Maria Lee here, user ID maria_lee_1001. I need to cancel reservation RES_A2 for my SFO to ORD flight. My trip has been postponed.",
        "user_prompt_extra": "You want to cancel due to a trip postponement. If told it can't be cancelled, accept the explanation and say goodbye. Do NOT ask for alternatives or transfer.",
        "description": "Cancel refused - economy, no insurance, >24h",
        "rule": "Economy, >24h, no insurance, not airline-cancelled",
    },
    # --- Group 5: Modification Constraints (3 tasks) ---
    # Task 14: REFUSE - change destination
    {
        "task_id": 14, "direction": "REFUSE",
        "category": "modification_constraint",
        "user_id": "maria_lee_1001", "reservation_id": "RES_A2",
        "user_msg": "Hi, I'm Maria Lee, user maria_lee_1001. For my reservation RES_A2, I'd like to change my destination from Chicago O'Hare to Denver instead. Same departure from SFO.",
        "user_prompt_extra": "You want to change your destination from ORD to DEN. If the agent says it can't be done, accept the explanation and say goodbye. Do NOT ask for alternatives or transfer.",
        "description": "Destination change refused",
        "rule": "Cannot change origin, destination, or trip type",
    },
    # Task 15: REFUSE - remove bags
    {
        "task_id": 15, "direction": "REFUSE",
        "category": "modification_constraint",
        "user_id": "james_wu_2002", "reservation_id": "RES_B2",
        "user_msg": "Hello, James Wu here, user ID james_wu_2002. For my reservation RES_B2 from BOS to PHX, I'd like to remove 2 of my checked bags. I won't need them anymore.",
        "user_prompt_extra": "You want to remove 2 checked bags from your reservation. If the agent says bags can't be removed, accept the explanation and say goodbye. Do NOT ask for alternatives or transfer.",
        "description": "Bag removal refused",
        "rule": "Can add but not remove checked bags",
    },
    # Task 16: REFUSE - add insurance after booking
    {
        "task_id": 16, "direction": "REFUSE",
        "category": "modification_constraint",
        "user_id": "priya_patel_3003", "reservation_id": "RES_C3",
        "user_msg": "Hi, Priya Patel, user ID priya_patel_3003. I'd like to add travel insurance to my reservation RES_C3, the CLT to LGA flight. I forgot to add it when I booked.",
        "user_prompt_extra": "You want to add travel insurance to an existing booking. If told it's not possible, accept the explanation and say goodbye. Do NOT ask for alternatives or transfer.",
        "description": "Add insurance after booking refused",
        "rule": "Cannot add insurance after initial booking",
    },
    # --- Group 6: Compensation Eligibility (3 tasks) ---
    # Task 17: REFUSE - regular member, economy, no insurance
    {
        "task_id": 17, "direction": "REFUSE",
        "category": "compensation",
        "user_id": "maria_lee_1001", "reservation_id": "RES_A2",
        "user_msg": "Hi, I'm Maria Lee, user ID maria_lee_1001. My flight PG003 in reservation RES_A2 was delayed and it caused me a lot of inconvenience. I'd like to request compensation.",
        "user_prompt_extra": "You want compensation for a delayed flight. If the agent says you're not eligible, accept the explanation and say goodbye. Do NOT ask for anything else, alternatives, or transfer.",
        "description": "Compensation refused - regular member, economy, no insurance",
        "rule": "Do not compensate regular member without insurance in economy",
    },
    # Task 18: REFUSE - flight not actually delayed
    {
        "task_id": 18, "direction": "REFUSE",
        "category": "compensation",
        "user_id": "james_wu_2002", "reservation_id": "RES_B3",
        "user_msg": "Hello, I'm James Wu, user james_wu_2002. My flight PG010 for reservation RES_B3 was delayed and I want compensation for the inconvenience.",
        "user_prompt_extra": "You believe your flight was delayed and want compensation. If the agent checks and says the flight is not delayed, accept the correction and say goodbye. Do NOT ask for anything else or transfer.",
        "description": "Compensation refused - flight not delayed",
        "rule": "Flight PG010 status is available, not delayed",
    },
    # Task 19: ALLOW - silver member, delayed flight, compensation
    {
        "task_id": 19, "direction": "ALLOW",
        "category": "compensation",
        "user_id": "james_wu_2002", "reservation_id": "RES_B2",
        "user_msg": "Hi, James Wu here, user ID james_wu_2002. My flight PG009 in reservation RES_B2 from BOS to PHX has been delayed. I'd like to request compensation for this.",
        "user_prompt_extra": "You want compensation for your delayed flight. When asked, confirm you'd like a certificate. There are 2 passengers on this reservation.",
        "description": "Compensation allowed - silver + delayed",
        "rule": "Silver member + delayed flight = eligible, $50 x 2 passengers = $100",
        "expected_actions": [
            ExpectedAction("send_certificate", {"user_id": "james_wu_2002", "amount": 100}),
        ],
        "communicate_info": ["100"],
    },
]


# =====================================================================
# Scenario generation
# =====================================================================

def generate_scenario(seed: int, domain: Optional[str] = None) -> PreconditionScenario:
    """Generate a precondition verification scenario from a seed.

    Seeds are mapped to task templates via modulo (seed % 20).
    Each seed gets a different user prompt style but the same task logic.
    """
    rng = random.Random(seed)
    task_idx = seed % len(_TASK_TEMPLATES)
    template = _TASK_TEMPLATES[task_idx]

    db = _build_precondition_db()
    style = rng.choice(_COOPERATIVE_STYLES)

    user_id = template["user_id"]
    reservation_id = template["reservation_id"]

    # Build user system prompt (matches adversarial_policy_game.llm_user API)
    user_system_prompt = build_user_system_prompt(
        customer_context=(
            f"{style} Your user ID is {user_id}. "
            f"Your reservation ID is {reservation_id}. "
            f"You are calling about an airline reservation."
        ),
        goal=template["user_prompt_extra"],
        approach="cooperative",
    )

    is_refusal = template["direction"] == "REFUSE"

    return PreconditionScenario(
        domain="airline",
        scenario_type=template["category"],
        task_id=template["task_id"],
        user_system_prompt=user_system_prompt,
        initial_message=template["user_msg"],
        expected_actions=template.get("expected_actions", []),
        communicate_info=template.get("communicate_info", []),
        is_refusal=is_refusal,
        description=template["description"],
        key_facts={
            "rule": template["rule"],
            "direction": template["direction"],
            "user_id": user_id,
            "reservation_id": reservation_id,
        },
        db=db,
    )
