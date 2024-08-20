# Makefile

# Export environment variables from .env

.PHONY: all setup build pull prep install-requirements

all: setup build pull prep

setup: install-requirements
	# Additional setup commands if needed

install-requirements:
	pip install -r requirements.txt

build: setup
	docker-compose up --build -d

pull: build
	docker-compose exec ollama ollama pull phi3

prep: pull
	docker-compose exec streamlit python prep.py
