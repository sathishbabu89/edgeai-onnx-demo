import os
import random
import numpy as np
from sklearn.ensemble import IsolationForest
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

MODEL_PATH = os.path.join(os.path.dirname(__file__), "edge_model.onnx")

def train_edge_model():
    # synthetic data
    data = [[random.uniform(1, 500), random.randint(0, 23), random.randint(0, 9)]
            for _ in range(1000)]
    
    clf = IsolationForest(random_state=42, contamination=0.05)
    clf.fit(data)

    # define input type
    initial_type = [("input", FloatTensorType([None, 3]))]

    # ✅ Force opset versions for ONNX + ONNX-ML
    onnx_model = convert_sklearn(
        clf,
        initial_types=initial_type,
        target_opset={
            "": 15,              # default ONNX domain opset
            "ai.onnx.ml": 3      # ONNX-ML opset (fixes IsolationForest bug)
        }
    )

    # save
    with open(MODEL_PATH, "wb") as f:
        f.write(onnx_model.SerializeToString())

    return MODEL_PATH


def edge_ai_check(tx, model_path=MODEL_PATH):
    import onnxruntime as rt

    sess = rt.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name

    features = np.array([[tx["amount"], int(tx["time"].split(":")[0]), tx["city_code"]]],
                        dtype=np.float32)
    pred = sess.run([label_name], {input_name: features})[0]
    return "Fraudulent" if pred[0] == -1 else "Legit"
