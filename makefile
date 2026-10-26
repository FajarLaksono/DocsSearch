-include .env

LLM_MODEL ?= llama3.2:3b
EMBED_MODEL ?= all-minilm

.PHONY: up down logs psql smoke models pull-models reset-db clean shell

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
