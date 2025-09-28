import random
from datetime import datetime

CITIES = ["London", "Manchester", "Birmingham", "Leeds", "Glasgow"]

def generate_paypoint_transaction():
    """Simulate a PayPoint barcode cash deposit transaction."""
    tx = {
        "type": "PayPoint Deposit",
        "amount": round(random.uniform(5, 500), 2),
        "city": random.choice(CITIES),
        "city_code": random.randint(0, 9),
        "time": datetime.now().strftime("%H:%M:%S")
    }
    return tx
