.PHONY: test run

PYTHON ?= python
PORT ?= 8000

test:
	PYTHONPATH=src $(PYTHON) -m unittest discover -s tests -v

run:
	cd src && $(PYTHON) -m uvicorn main:app --host 127.0.0.1 --port $(PORT) --workers 1
