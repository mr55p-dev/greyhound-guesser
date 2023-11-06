from flask import Flask, request
import numpy as np
import werkzeug

labels = ["odds", "distance", "finish"]
fields = [f"dog-{i}-{j}" for i in range(5) for j in labels] 
fields.append("race-length")

app = Flask(__name__)

class BadFormError(werkzeug.exceptions.HTTPException):
    code = 400
    description = "Form is missing a required property"


@app.route("/predict", methods=["POST"])
def predict():
    form_fields = {}
    print(fields)
    for field in fields:
        if field not in request.form:
            raise BadFormError()
        form_fields[field] = request.form[field]

    arr = np.zeros((19, 1), dtype=np.float32)
    arr[0, 0] = form_fields["race-length"]
    for idx, field in enumerate(fields):
        arr[1 + idx, 0] = form_fields[field]

    print(arr)

    return {
        0: 0.1,
        1: 0.0,
        2: 0.0,
        3: 0.9,
        4: 0.0,
        5: 0.0,
    }
