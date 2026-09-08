.PHONY: install run run-mock test lint docker-build compose-up compose-down k8s-dev k8s-prod

install:
	pip install -r requirements-dev.txt

run:
	uvicorn app.main:app --reload --port 8000

run-mock:
	LLM_MOCK=true uvicorn app.main:app --reload --port 8000

test:
	pytest -v

lint:
	ruff check .

docker-build:
	docker build -t ai-platform-starter:local .

compose-up:
	docker compose up --build -d

compose-down:
	docker compose down

k8s-dev:
	kubectl apply -k k8s/overlays/dev

k8s-prod:
	kubectl apply -k k8s/overlays/prod
