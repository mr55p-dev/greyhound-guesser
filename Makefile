LATEST_MODEL := $(shell echo "models/$(shell ls -Art models | tail -n 1)")
init:
	python3 -m venv venv
	venv/bin/python -m pip install -r requirements.txt

get-latest:
	@echo $(LATEST_MODEL)

clean-venv:
	rm -rf venv

lab:
	venv/bin/python -m jupyter lab

build:
	go buid

run:
	INFERENCE_HOSTNAME=localhost:5000 go run ./bin/server

model-server:
	echo $(LATEST_MODEL)
	FLASK_DEBUG=1 \
	FLASK_APP=./src/server \
	TORCH_WEIGHT_PATH=$(LATEST_MODEL) \
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

build-platform-images:
	docker build -t public.ecr.aws/n7b0m3g0/greyhound-guesser/server:amd64 -f server.Dockerfile --platform linux/amd64 .
	docker build -t public.ecr.aws/n7b0m3g0/greyhound-guesser/inference:amd64 -f inference.Dockerfile --platform linux/amd64 .
	docker image push public.ecr.aws/n7b0m3g0/greyhound-guesser/server:amd64 
	docker image push public.ecr.aws/n7b0m3g0/greyhound-guesser/inference:amd64 

test-inference:
	curl --location 'http://127.0.0.1:5000/predict' \
	--header 'Content-Type: application/x-www-form-urlencoded' \
	--data-urlencode 'dog-0-odds=1' \
	--data-urlencode 'dog-0-finished=1' \
	--data-urlencode 'dog-0-distance=2' \
	--data-urlencode 'dog-1-odds=1' \
	--data-urlencode 'dog-1-finished=1' \
	--data-urlencode 'dog-1-distance=1' \
	--data-urlencode 'dog-2-odds=1' \
	--data-urlencode 'dog-2-finished=1' \
	--data-urlencode 'dog-2-distance=1' \
	--data-urlencode 'dog-3-odds=1' \
	--data-urlencode 'dog-3-finished=1' \
	--data-urlencode 'dog-3-distance=1' \
	--data-urlencode 'dog-4-odds=1' \
	--data-urlencode 'dog-4-finished=1' \
	--data-urlencode 'dog-4-distance=1' \
	--data-urlencode 'dog-5-odds=1' \
	--data-urlencode 'dog-5-finished=1' \
	--data-urlencode 'dog-5-distance=1' \
	--data-urlencode 'race-length=0.5'

air:
	INFERENCE_HOSTNAME=localhost:5000 air

.PHONY = init clean-venv lab build run model-archive torch-server
