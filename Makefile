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
	INFERENCE_HOSTNAME=localhost:5000 go run ./bin/server

model-server:
	FLASK_DEBUG=1 \
	FLASK_APP=./src/server \
	TORCH_WEIGHT_PATH=./models/gg-2023-11-09_16-31.pt \
	venv/bin/python3 -m flask run \
	--host=localhost

train:
	venv/bin/python3 train.py

docker-build-inference:
	docker build \
		--build-arg DEBUG=1 \
		--build-arg \
		-f inference.Dockerfile \
		-t gg-inference \
		.

docker-build-server:
	docker build \
		-f server.Dockerifle \
		-t gg-server \
		.

.PHONY = init clean-venv lab build run model-archive torch-server
