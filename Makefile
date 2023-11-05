venv:
	python3 -m venv venv
	venv/bin/python -m pip install -r requirements.txt

clean-venv:
	rm -rf venv

lab:
	venv/bin/python -m jupyter lab

.PHONY = venv clean-venv lab
