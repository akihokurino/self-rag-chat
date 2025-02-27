MAKEFLAGS=--no-builtin-rules --no-builtin-variables --always-make
ROOT := $(realpath $(dir $(lastword $(MAKEFILE_LIST))))
SHELL := /bin/bash

vendor:
	source venv/bin/activate && poetry install

update-modules:
	source venv/bin/activate && poetry update

types:
	source venv/bin/activate && mypy .

run-db:
	docker-compose up

run-server:
	source venv/bin/activate && python -m server

run-create-index:
	source venv/bin/activate && python -m create_index

gen-migration-file:
	source venv/bin/activate && alembic revision --autogenerate -m "auto"

migrate:
	poetry run alembic upgrade head

check-migration:
	poetry run alembic check