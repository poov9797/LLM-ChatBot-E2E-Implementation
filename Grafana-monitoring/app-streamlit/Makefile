# Makefile

# Step 1: Install Python dependencies
install:
	@echo "Installing Python dependencies..."
	pip install -r requirements.txt
	@sleep 5

# Step 2: Build and start the Docker containers
up:
	@echo "Building and starting Docker containers..."
	docker-compose up --build -d
	@sleep 5

# Step 3: Pull the "phi3" model using Docker
pull_phi3:
	@echo "Pulling the phi3 model..."
	docker-compose exec ollama ollama pull phi3
	@sleep 5

# Step 4: Set environment variable POSTGRES_HOST
set_env:
	@echo "Setting environment variable POSTGRES_HOST to 'localhost'..."
	export POSTGRES_HOST="localhost"
	@sleep 5

# Step 5: Run the Python preparation script
prep:
	@echo "Running prep.py..."
	python prep.py
	@sleep 5

# Combine steps 1-5 sequentially, with a 5-second wait between each step
all: install up pull_phi3 set_env prep
	@echo "All steps completed!"

.PHONY: install up pull_phi3 set_env prep all
