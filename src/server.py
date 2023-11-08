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
        network: Network = torch.load("./models/gg-2023-11-07_18-24.pt")
        self.model = network

    def predict(self, form):
        input = np.zeros((19, 1), dtype=np.float32)
        input[0, 0] = form["race-length"]
        for idx, field in enumerate(fields):
            input[1 + idx, 0] = form[field]

        print("Decoded tensor")
        print(input)

        input = torch.tensor(input, device=self.device)
        pred = self.model.forward(input.T)
        return pred

labels = ["odds", "distance", "finished"]
fields = [f"dog-{i}-{j}" for i in range(5) for j in labels] 
fields.append("race-length")

model = Model()
app = Flask(__name__)
app.logger.info("Started app")

@app.route("/predict", methods=["POST"])
def predict():
    form_fields = {}
    app.logger.debug(f"received form request with fields: \n{request.form}")
    for field in fields:
        if field not in request.form:
            print("Bad field")
            print(field)
            raise BadFormError()
        form_fields[field] = request.form[field]

    print("Received request with form: ")
    pprint(form_fields)

    out = model.predict(form_fields)
    print("Generated predictions")
    pprint(out)
    return out[0, :].tolist()

