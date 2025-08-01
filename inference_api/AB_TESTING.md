# A/B Testing Infrastructure

This document describes the A/B testing infrastructure for the Churn Prediction API, including implementation details and usage.

## Architecture

```
┌─────────────┐     ┌─────────────────────┐     ┌─────────────────┐
│   Client    ├────►│       NGINX         │     │  API Instances  │
└─────────────┘     │  - Load Balancer    ├────►│  - Production   │
                    │  - Traffic Splitting │     │  - Shadow       │
                    └──────────┬──────────┘     └────────┬────────┘
                               │                         │
                         ┌─────▼─────┐           ┌───────▼───────┐
                         │  Logging  │           │  A/B Test     │
                         │  Service  │◄─────────►│  Evaluator    │
                         └───────────┘           └───────────────┘
```

## Components

### 1. NGINX Load Balancer (`nginx/nginx.conf`)
- **Traffic Splitting**: 50/50 split between production and shadow models
- **Sticky Sessions**: Uses client IP + User-Agent for consistent routing
- **Logging**: Custom log format includes A/B test group information
- **Health Checks**: Built-in health check endpoint at `/health`

### 2. API Endpoints
- **Production**: Serves `random_forest_RF_best` model
- **Shadow**: Serves `logistic_regression_LR_second_best` model
- **Core Endpoints**:
  - `POST /api/v1/predict` - Get predictions
  - `POST /api/v1/predict/update-ground-truth` - Submit ground truth
  - `GET /api/v1/management/ab-test/summary` - View test metrics
  - `GET /api/v1/management/ab-test/logs` - View prediction logs

### 3. A/B Test Evaluation
- **Real-time Metrics**:
  - Accuracy per model
  - Prediction times
  - Traffic distribution
- **Automatic Recommendations**: Suggests model promotion when statistically significant improvements are detected

## Getting Started

### 1. Start Services
```bash
docker-compose up --build -d
```

### 2. Verify Services
```bash
docker-compose ps
```

### 3. Access Endpoints
- **API Base**: `http://localhost:8083/api/v1`
- **A/B Test Dashboard**: `http://localhost:8083/api/v1/management/ab-test/summary`
- **API Documentation**: `http://localhost:8083/docs`

## Managing A/B Tests

### 1. Check Current Status
```bash
curl -s "http://localhost:8083/api/v1/management/ab-test/summary" | jq
```

### 2. Submit Ground Truth
```bash
# Format: /update-ground-truth?request_id=ID&ground_truth=[0|1]
curl -X POST "http://localhost:8083/api/v1/predict/update-ground-truth?request_id=123&ground_truth=1"
```

### 3. View Prediction Logs
```bash
# Get all logs
curl -s "http://localhost:8083/api/v1/management/ab-test/logs" | jq

# Filter logs missing ground truth
curl -s "http://localhost:8083/api/v1/management/ab-test/logs?missing_ground_truth=true" | jq
```

## Testing the Setup

### 1. Run Test Predictions
```bash
# Using the test script
python scripts/test_ab_testing.py

# Or manually:
curl -X POST "http://localhost:8083/api/v1/predict" \
  -H "Content-Type: application/json" \
  -d '{"SeniorCitizen":0, "tenure":12, "MonthlyCharges":70.5, ...}'
```

### 2. Check Metrics
```bash
# Get summary
curl -s "http://localhost:8083/api/v1/management/ab-test/summary" | jq

# View raw logs
curl -s "http://localhost:8083/api/v1/management/ab-test/logs" | jq
```

### 3. Evaluate Model Performance
```bash
# Run evaluation script
python scripts/evaluate_ab_test.py
```

## Configuration

### 1. NGINX (`nginx/nginx.conf`)
- **Traffic Split**: Adjust the `split_clients` block
  ```nginx
  split_clients "${remote_addr}${http_user_agent}${time_iso8601}" $ab_test_group {
      30%     "production";  # 30% traffic
      *       "shadow";       # 70% traffic
  }
  ```
- **Upstream Servers**: Configure in `upstream` blocks
- **Logging**: Customize `log_format` as needed

### 2. A/B Test Service (`app/services/ab_testing.py`)
- **Log Location**: `logs/ab_test_logs/weekly_ab_test_logs.json`
- **Metrics**: Customize in `get_test_summary()`
- **Evaluation**: Adjust statistical thresholds in `get_model_recommendation()`

### 3. Environment Variables (`.env`)
```env
# Traffic split between models (0.0 to 1.0)
TRAFFIC_SPLIT=0.5

# Model versions
PRODUCTION_MODEL=random_forest_RF_best
SHADOW_MODEL=logistic_regression_LR_second_best
```

## Monitoring & Maintenance

### 1. Logs
```bash
# API Logs
docker-compose logs -f inference-api

# NGINX Access Logs
docker-compose exec nginx tail -f /var/log/nginx/access.log

# A/B Test Logs
tail -f logs/ab_test_logs/*.json
```

### 2. Key Metrics
- **Accuracy**: Model prediction accuracy (%)
- **Latency**: Average prediction time (ms)
- **Traffic**: Requests per model
- **Ground Truth Coverage**: % of predictions with ground truth

### 3. Automated Alerts
Set up monitoring for:
- Significant performance degradation
- Ground truth coverage below threshold
- Model drift detection

## Troubleshooting

### Common Issues
1. **Missing Ground Truth**
   - Symptom: Low ground truth coverage in metrics
   - Fix: Submit missing ground truth using the update endpoint

2. **NGINX Routing Issues**
   - Check logs: `docker-compose logs nginx`
   - Verify ports in `docker-compose.yml`

3. **Model Performance**
   - Check for data drift
   - Verify feature preprocessing
   - Review ground truth quality

4. **Log Rotation**
   - Logs are rotated weekly
   - Old logs are compressed and archived
   - Check port 80 is available
   - Verify `nginx/nginx.conf` is valid

2. **No traffic to Group B**:
   - Check NGINX logs: `docker-compose logs nginx`
   - Verify shadow API is running: `docker-compose ps`

3. **Missing ground truth data**:
   - Ensure the update-ground-truth endpoint is being called
   - Check logs for errors

## Cleanup

To stop and remove all containers and volumes:

```bash
docker-compose down -v
```
