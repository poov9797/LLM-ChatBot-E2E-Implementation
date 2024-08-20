# Makefile

.PHONY: all setup build pull prep

all: setup  build pull prep

build: setup
	docker-compose up --build -d

pull: build
	docker-compose exec ollama ollama pull gemma2:2b

prep: pull
	docker-compose exec streamlit python prep.py
