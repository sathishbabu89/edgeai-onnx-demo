import streamlit as st
import time
from dotenv import load_dotenv
import os
from openai import OpenAI
import sys

# ---------------------------
# Fix Python imports for sibling packages
# ---------------------------
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.edge_model import train_edge_model, edge_ai_check
from utils.transaction_gen import generate_paypoint_transaction

# ---------------------------
# Setup Streamlit
# ---------------------------
st.set_page_config(page_title="PayPoint EdgeAI Demo", layout="wide")
load_dotenv()

# ---------------------------
# Configure Grok CloudAI
# ---------------------------
client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)
# ✅ Correct model string for OpenAI SDK
GROK_MODEL = "deepseek/deepseek-chat-v3.1:free"
# ---------------------------
# Train / Load EdgeAI model
# ---------------------------
model_path = train_edge_model()
st.info(f"✅ EdgeAI ONNX model ready at: {model_path}")

# ---------------------------
# CloudAI reasoning
# ---------------------------
def cloud_ai_reasoning(tx, edge_result):
    prompt = f"""
    You are a fraud analyst for Lloyds Bank.
    Transaction details:
    - Type: {tx['type']}
    - Amount: £{tx['amount']}
    - City: {tx['city']}
    - Time: {tx['time']}

    Edge AI classified this transaction as: {edge_result}.
    Explain briefly (2 sentences max) why it might be {edge_result}.
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
# Streamlit UI
# ---------------------------
st.title("🏧 PayPoint EdgeAI Fraud Detection")
st.write("Simulate **real-time fraud detection** for PayPoint barcode deposits using EdgeAI ONNX + Grok CloudAI explanations.")

if st.button("🔄 Generate New Deposit"):
    tx = generate_paypoint_transaction()
    st.subheader("📥 Incoming Deposit")
    st.json(tx)

    # EdgeAI ONNX Result
    with st.spinner("⚡ EdgeAI (ONNX) analyzing..."):
        time.sleep(0.5)
        edge_result = edge_ai_check(tx, model_path)
    st.success(f"⚡ EdgeAI Verdict: {edge_result}")

    # CloudAI Reasoning
    with st.spinner("☁️ Cloud AI analyzing with Grok..."):
        time.sleep(2)
        reasoning = cloud_ai_reasoning(tx, edge_result)
    st.info(f"☁️ Grok says: {reasoning}")
