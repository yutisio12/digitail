# Digital Meter OCR API

API for reading digital meter values from images using OCR.

## Stack

- **Spring Boot 3.x** (Java 17) — REST API, auth, orchestration
- **Python 3.10 + FastAPI** — OCR processing with EasyOCR
- **Docker** — container orchestration

## Quick Start

```bash
# Start all services
docker-compose up --build

# Or run Python OCR service locally
cd python-ocr
pip install -r requirements.txt
uvicorn main:app --reload --port 8001
```

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `API_KEYS` | `dev-key-123,test-key-456` | Comma-separated valid API keys |

## Usage

```bash
curl -X POST http://localhost:8080/api/v1/meter/read \
  -H "X-API-KEY: prod-key-xxx" \
  -F "image=@meter_test.jpg" \
  -F "meter_type=electric"
```

## API Endpoints

| Method | Path | Auth | Description |
|---|---|---|---|
| POST | `/api/v1/meter/read` | X-API-KEY | Upload meter image for OCR |
| GET | `/api/v1/health` | No | Health check |
