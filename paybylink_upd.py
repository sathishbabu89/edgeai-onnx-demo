import streamlit as st
import time
from dotenv import load_dotenv
import os
from openai import OpenAI
import sys
import random
import numpy as np
from datetime import datetime
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as rt

# ---------------------------
# Fix Python imports for sibling packages
# ---------------------------
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------------------------
# Setup Streamlit
# ---------------------------
st.set_page_config(page_title="PayByLink Fraud Detection", layout="wide")
load_dotenv()

# ---------------------------
# Configure Cloud AI (Grok or other) for Reasoning
# ---------------------------
client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)
GROK_MODEL = "meta-llama/llama-3.3-8b-instruct:free"

# ---------------------------
# Transaction Data Generation for Pay by Link
# ---------------------------
CITIES = ["London", "Manchester", "Birmingham", "Leeds", "Glasgow"]
PAYMENT_METHODS = ["Credit Card", "Debit Card", "PayPal", "Wallet"]

REFERENCES = ["Yesterday dinner", "Loan repayment", "Gift for Mom", "Rent payment", "Shared travel expenses"]

def generate_paylink_transaction():
    """Simulate a Pay by Link transaction (pre-payment)."""
    reference = random.choice(REFERENCES)
    amount = round(random.uniform(10, 500), 2)  # Amount between 10 and 500 Pounds

    # Assigning city and city_code
    city = random.choice(CITIES)
    city_code = CITIES.index(city)  # This assigns a unique code to each city based on its index

    tx = {
        "reference": reference,
        "amount": amount,
        "device_id": random.randint(1000, 9999),  # Unique device ID for Payee
        "city": city,
        "city_code": city_code,  # Added city_code
        "time": datetime.now().strftime("%H:%M:%S")  # This is when the request is generated, not the actual payment time
    }

    # Simulate fraud detection (e.g., if amount is unusually high or in odd hours)
    if random.random() < 0.2:  # 20% chance to simulate a fraud case
        tx['amount'] = round(random.uniform(1000, 5000), 2)  # Very high amount
        tx['time'] = f"{random.randint(0, 2)}:{random.randint(0, 59)}:{random.randint(0, 59)}"  # Odd time, late night or early morning

    return tx



# ---------------------------
# Edge AI Model Training (IsolationForest)
# ---------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "edge_model.onnx")

def train_edge_model():
    """Train the Edge AI model."""
    data = [[random.uniform(10, 500), random.randint(0, 23), random.randint(0, 9)] for _ in range(1000)]
    
    from sklearn.ensemble import IsolationForest
    # Increase contamination to 0.5 to force more fraud detection
    clf = IsolationForest(random_state=42, contamination=0.5)
    clf.fit(data)

    initial_type = [("input", FloatTensorType([None, 3]))]

    onnx_model = convert_sklearn(clf, initial_types=initial_type, target_opset={"": 15, "ai.onnx.ml": 3})

    with open(MODEL_PATH, "wb") as f:
        f.write(onnx_model.SerializeToString())

    return MODEL_PATH

def edge_ai_check(tx, model_path=MODEL_PATH):
    """Detect fraud using Edge AI model."""
    sess = rt.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name

    # Adjusted decision threshold for fraud detection
    features = np.array([[tx["amount"], int(tx["time"].split(":")[0]), tx["city_code"]]], dtype=np.float32)
    pred = sess.run([label_name], {input_name: features})[0]
    
    # Adjust logic to be more sensitive to fraud (increase fraud detection)
    return "Fraudulent" if pred[0] == -1 else "Legit"

# ---------------------------
# Cloud AI Reasoning (explanation)
# ---------------------------
def cloud_ai_reasoning(tx, edge_result):
    prompt = f"""
    You are a fraud analyst for Pay by Link transactions.
    Transaction details:
    - Reference: {tx['reference']}
    - Amount: £{tx['amount']}
    - City: {tx['city']}
    - Time: {tx['time']}

    Edge AI classified this transaction as: {edge_result}.
    Explain briefly why it might be {edge_result}.
    """
    try:
        resp = client.chat.completions.create(
            model=GROK_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"(Cloud AI reasoning failed: {e})"


# ---------------------------
# Main Streamlit UI
# ---------------------------
st.markdown(""" 
    <style>
        .title {
            font-size: 32px;
            font-weight: bold;
            color: #3E5C76;
        }
        .subheader {
            color: #3E5C76;
            font-size: 22px;
            margin-top: 20px;
        }
        .card {
            background-color: #F0F4F8;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.1);
        }
        .button {
            background-color: #1E88E5;
            color: white;
            font-size: 16px;
            border-radius: 5px;
            padding: 10px 20px;
            cursor: pointer;
            transition: background-color 0.3s;
        }
        .button:hover {
            background-color: #1565C0;
        }
        .info-box {
            border-left: 5px solid #1E88E5;
            padding-left: 15px;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="title">🏧 PayByLink Fraud Detection Using Edge AI</p>', unsafe_allow_html=True)
st.write("Simulate **real-time fraud detection** for PayByLink transactions using Edge AI and Cloud AI explanations.")
st.image("linkpay.png", caption="PayByLink Overview", width='stretch')
# Check if the transaction data is already in session state
if 'tx' not in st.session_state:
    st.session_state['tx'] = None

# Button to generate new Pay by Link transaction
if st.button("🔄 Generate New Pay by Link Transaction"):
    # Generate a new transaction and store it in session state
    tx = generate_paylink_transaction()
    st.session_state['tx'] = tx

if st.session_state['tx']:
    tx = st.session_state['tx']
    st.markdown(f'<div class="subheader">📥 Incoming Transaction</div>', unsafe_allow_html=True)
    st.json(tx)

    # EdgeAI ONNX Result
    with st.spinner("⚡ EdgeAI (ONNX) analyzing..."):
        time.sleep(0.5)
        edge_result = edge_ai_check(tx, MODEL_PATH)
    st.success(f"⚡ EdgeAI Verdict: {edge_result}")

    # CloudAI Reasoning
    with st.spinner("☁️ Cloud AI analyzing with Grok..."):
        time.sleep(2)
        reasoning = cloud_ai_reasoning(tx, edge_result)
    st.info(f"☁️ Meta Llama says: {reasoning}", icon="ℹ️")
