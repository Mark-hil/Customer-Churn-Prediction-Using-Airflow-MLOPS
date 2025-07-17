# Churn Prediction Inference API

A FastAPI-based service for making churn predictions using trained models from your Airflow pipeline.

## Features

- Loads the latest trained model from the specified directory
- Provides a RESTful API for making predictions
- Includes health check endpoint
- Containerized with Docker for easy deployment
- Automatic model reloading when new models are available

## Prerequisites

- Docker and Docker Compose
- Trained model files in the `data/models` directory

## Getting Started

1. **Build and start the service**

   ```bash
   docker-compose up --build -d
   ```

2. **Verify the service is running**

   ```bash
   curl http://localhost:8000/api/v1/health
   ```

   Should return:
   ```json
   {"status": "healthy", "model_loaded": true}
   ```

3. **Make a prediction**

   ```bash
   curl -X POST "http://localhost:8000/api/v1/predict" \
        -H "Content-Type: application/json" \
        -d '{"feature1": value1, "feature2": value2, ...}'
   ```

## API Endpoints

- `GET /` - API information
- `GET /api/v1/health` - Health check
- `POST /api/v1/predict` - Make a prediction
- `GET /api/v1/docs` - Interactive API documentation (Swagger UI)
- `GET /api/v1/redoc` - Alternative API documentation (ReDoc)

## Configuration

Environment variables can be set in the `docker-compose.yml` file or in a `.env` file:

- `MODEL_DIR`: Directory containing model files (default: `/models`)
- `PORT`: Port to run the API on (default: `8000`)
- `DEBUG`: Enable debug mode (default: `True` in development)

## Development

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Run the development server:

   ```bash
   uvicorn app.main:app --reload
   ```

3. The API will be available at `http://localhost:8000`

## Deployment

For production deployment, consider:

1. Setting `DEBUG=False`
2. Configuring a proper WSGI server like Gunicorn with Uvicorn workers
3. Setting up proper logging and monitoring
4. Implementing authentication/authorization
5. Setting up HTTPS with a reverse proxy like Nginx
