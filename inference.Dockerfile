FROM python:3.11

WORKDIR /app

COPY requirements-inference.txt requirements.txt
RUN python3 -m pip install -r requirements.txt

COPY src/*.py src/

EXPOSE 5000

ARG DEBUG=0

ENV FLASK_APP=./src/server
ENV FLASK_DEBUG=0

CMD [ "python3", "-m" , "flask", "run", "--host=inference", "--port=5000"]
