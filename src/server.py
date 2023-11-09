import os
from flask import Flask, request
import numpy as np
import werkzeug
import torch
from pprint import pprint

from src.utils import get_device
from src.model import Network

class BadFormError(werkzeug.exceptions.HTTPException):
    code = 400
    description = "Form is missing a required property"

class Model():
    def __init__(self):
        self.device = get_device()
        
        weight_path = os.getenv("TORCH_WEIGHT_PATH")
        if not weight_path:
            raise OSError("TORCH_WEIGHT_PATH environment variable is missing")

        network = Network()
        network.load_state_dict(torch.load(weight_path, map_location=self.device))
        network.to(self.device)
        self.model = network

    def predict(self, form):
        input = np.zeros((19, 1), dtype=np.float32)
        for idx, field in enumerate(fields):
            input[idx, 0] = form[field]

        print("Decoded tensor")
        print(input)

        input = torch.tensor(input, device=self.device)
        pred = self.model.forward(input.T)
        return pred

labels = ["odds", "distance", "finished"]
fields = ["race-length"] + [f"dog-{i}-{j}" for i in range(5) for j in labels] 

model = Model()
app = Flask(__name__)
app.logger.info("Started app")

@app.route("/health", methods=["GET"])
def health():
    return "ok"

@app.route("/predict", methods=["POST"])
def predict():
    form_fields = {}
    app.logger.debug(f"received form request with fields: \n{request.form}")
    for field in fields:
        if field not in request.form:
            app.logger.debug("Bad field")
            app.logger.debug(field)
            raise BadFormError()
        form_fields[field] = request.form[field]

    out = model.predict(form_fields)
    app.logger.debug("Generated predictions")
    app.logger.debug(out)
    return out[0, :].tolist()

