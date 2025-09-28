EdgeAI PayPoint Fraud Detection Demo
📌 Overview

This project simulates how Edge AI can be applied to PayPoint barcode deposits for fraud detection.
Instead of sending all transactions to the cloud, an ONNX-based anomaly detection model runs locally to provide real-time classification (Fraudulent / Legit).
Additionally, a cloud-based LLM (Grok) provides reasoning to explain the decision in natural language.

🚀 Features

Simulates a PayPoint deposit workflow

Edge AI fraud detection with IsolationForest converted to ONNX

Streamlit app for interactive UI

Hybrid reasoning: Edge ML for decision, Cloud LLM for explanation

🏗️ Tech Stack

Python 3.10+

scikit-learn

skl2onnx

onnxruntime

Streamlit

OpenRouter (Grok LLM)

📂 Project Structure
edgeai-paypoint-demo/
│── app/
│   └── paypoint_edgeai_app.py       # Streamlit app
│── models/
│   └── edge_model.py                # Train & convert model to ONNX
│── utils/
│   └── cloud_ai.py                  # Grok LLM reasoning helper
│── requirements.txt
│── README.md

🔧 Installation
git clone https://github.com/<your-username>/edgeai-paypoint-demo.git
cd edgeai-paypoint-demo
python -m venv venv
source venv/bin/activate   # (Windows: venv\Scripts\activate)
pip install -r requirements.txt

▶️ Run the Demo
streamlit run app/paypoint_edgeai_app.py

💡 Example Use Case

Alice generates a barcode in her Lloyds banking app

She visits a PayPoint shop and deposits £50

Edge AI at PayPoint instantly classifies the transaction

If flagged as suspicious, the cloud LLM (Grok) provides reasoning:
“The deposit was unusually high for the time of day and location, suggesting possible mule activity.”

🔮 Future Enhancements

Integration with real PayPoint APIs

Richer fraud signals (device, location, velocity)

Model updates pushed securely from the cloud

⚠️ Disclaimer

This project is for educational/demo purposes only and not production-grade fraud detection.
