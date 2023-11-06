init:
	python3 -m venv venv
	venv/bin/python -m pip install -r requirements.txt

clean-venv:
	rm -rf venv

lab:
	venv/bin/python -m jupyter lab

build:
	go buid

run:
	go run ./bin/server

model-server:
	venv/bin/python3 -m flask --app ./src/server run

.PHONY = init clean-venv lab build run model-archive torch-server
