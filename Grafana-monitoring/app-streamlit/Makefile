# Makefile

.PHONY: all setup build pull prep

all: setup  build pull prep

build: setup
	docker-compose up --build -d

pull: build
	docker-compose exec ollama ollama pull phi3

prep: pull
	docker-compose exec streamlit python prep.py
