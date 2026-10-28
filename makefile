-include .env

LLM_MODEL ?= llama3.2:3b
EMBED_MODEL ?= all-minilm

.PHONY: up down logs psql smoke ingest reingest retrieve ask trace eval agent test models pull-models reset-db clean shell

up:            ## start Postgres and the app (build if needed)
	docker compose up -d --build

down:          ## stop everything; data is kept
	docker compose down

logs:          ## follow the app logs
	docker compose logs -f app

psql:          ## open a SQL shell on the database
	docker compose exec db psql -U aidocsuser -d aidocs

smoke:         ## prove db + chat model + embedding model all work
	docker compose run --rm app python -m src.smoke

test:          ## run unit tests (no database, no model)
	docker compose run --rm --no-deps app python -m pytest -q

models:        ## list the models installed in Ollama (host)
	ollama list

pull-models:   ## download the models named in .env
	ollama pull $(LLM_MODEL)
	ollama pull $(EMBED_MODEL)

reset-db:      ## delete the database volume and recreate the schema
	docker compose down
	docker volume rm docssearch_pgdata || echo "volume already gone"
	docker compose up -d

clean:         ## stop everything and delete the database volume
	docker compose down -v

shell:         ## a bash shell inside the app container
	docker compose exec app bash

ingest:        ## load, chunk, embed, and store everything under data/
	docker compose run --rm app python -m src.ingest

reingest:      ## same, but clear the database first (use after deleting files)
	docker compose run --rm app python -m src.ingest --reset

retrieve:      ## show the top-k chunks for a question, no generation
	docker compose run --rm app python -m src.retrieve "$(Q)"

ask:           ## answer a question with citations
	docker compose run --rm app python -m src.ask "$(Q)"

trace:         ## print a stored run (latest, or ID=n)
	docker compose run --rm app python -m src.trace $(ID)

eval:          ## run the gold set (ARGS=--limit 5 --no-judge)
	docker compose run --rm app python -m src.evaluate $(ARGS)

agent:         ## run the agent loop (Q=..., ARGS=--thinking prompted)
	docker compose run --rm app python -m src.agent "$(Q)" $(ARGS)

explain:       ## explain an answer (Q=..., or ARGS=--run 12)
	docker compose run --rm app python -m src.explain "$(Q)" $(ARGS)

sweep:         ## run the experiment matrix (ARGS=--only k --dry-run)
	docker compose run --rm app python -m src.sweep $(ARGS)

report:        ## write reports/results.md from eval_results
	docker compose run --rm app python -m src.report $(ARGS)
