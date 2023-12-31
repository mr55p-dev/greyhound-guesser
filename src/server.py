import os
from flask import Flask, request
import numpy as np
import werkzeug
import torch

from src.utils import get_device
from src.model import Network


class BadFormError(werkzeug.exceptions.HTTPException):
    code = 400
    description = "Form is missing a required property"


class Model:
    def __init__(self):
        self.device = get_device()

        weight_path = os.getenv("TORCH_WEIGHT_PATH")
        if not weight_path:
            raise OSError("TORCH_WEIGHT_PATH environment variable is missing")

        network = Network()
        network.load_state_dict(torch.load(weight_path, map_location=self.device))
        network.to(self.device)
        self.model = network
        self.model_name = weight_path

    def sanitize(self, form):
        for field in form:
            if "odds" in field:
                form[field] = 1 / float(form[field])
            elif "distance" in field:
                form[field] = int(form[field]) / 1000
            elif "finished" in field:
                form[field] = (6 - int(form[field])) / 5
            else:
                form[field] = float(form[field])
        return form

    def predict(self, form) -> np.ndarray:
        form = self.sanitize(form)
        input = np.zeros((19, 1), dtype=np.float32)
        for idx, field in enumerate(fields):
            if type(form[field]) != float:
                raise ValueError(
                    f"Field {field} has invalid type {type(form[field])} (val: {form[field]})"
                )
            input[idx, 0] = form[field]

        input = torch.tensor(input, device=self.device)
        pred = self.model.forward(input.T).detach().cpu().numpy()
        app.logger.info("Generated prediction")
        app.logger.info(pred)
        return pred


labels = ["odds", "distance", "finished"]
fields = ["race-length"] + [f"dog-{i}-{j}" for i in range(5) for j in labels]

model = Model()
app = Flask(__name__)
app.logger.info("Started app")


@app.route("/health", methods=["GET"])
def health():
    return f"Ok. Using model `{model.model_name}`."


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

    out = model.predict(form_fields).reshape(-1)
    app.logger.debug("Generated predictions")
    app.logger.debug(out)

    best = out.argmax().item()
    confidence = out[best].item()
    app.logger.info(f"Responding to prediction with {best} confidence: {confidence}")

    return {
        "position": best,
        "confidence": confidence,
    }
