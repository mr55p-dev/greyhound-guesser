FROM python:3.11

WORKDIR /app

COPY requirements.txt requirements.txt
RUN python3 -m pip install -r requirements.txt

COPY . .

EXPOSE 5000


ARG DEBUG=0
ARG TORCH_WEIGHT_PATH

ENV FLASK_APP=./src/server
ENV FLASK_DEBUG=0
ENV TORCH_WEIGHT_PATH=${TORCH_WEIGHT_PATH}

CMD [ "python3", "-m" , "flask", "run", "--host=inference", "--port=5000"]
