#!/bin/bash
cd Grafana-monitoring/app-streamlit
sleep 5

export $(grep -v '^#' .env | xargs)
sleep 5

pip install -r requirements.txt
sleep 5

docker-compose up --build -d
sleep 5

docker-compose exec ollama ollama pull llama3.2
sleep 5

export POSTGRES_HOST="localhost"
sleep 5

python prep.py
