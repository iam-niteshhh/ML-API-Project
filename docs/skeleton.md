## THIS IS THE FOLDER STRUCTURE

#### ml-intelligence-api/
- app/
  - api/              &emsp;&emsp;&emsp;&emsp; # FastAPI routes (text.py, image.py)
  - core/             &emsp;&emsp;&emsp;&emsp; # Configs, constants
  - models/           &emsp;&emsp;&emsp;&emsp; # ML models or schemas
  - services/         &emsp;&emsp;&emsp;&emsp; # Logic to load/predict from models
- data/               &emsp;&emsp;&emsp;&emsp; # Sample text/images for testing
- tests/              &emsp;&emsp;&emsp;&emsp; # Pytest tests
- docker/             &emsp;&emsp;&emsp;&emsp; # Docker config
- requirements.txt
- Dockerfile
- README.md
- main.py             &emsp;&emsp;&emsp;&emsp; # FastAPI entrypoint


#### core flow:
User → FastAPI (/predict-text, /predict-image) → ML model → return result
