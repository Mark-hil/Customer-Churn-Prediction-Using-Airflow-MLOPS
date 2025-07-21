# A/B Testing Infrastructure

This document describes the A/B testing infrastructure for the Churn Prediction API.

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Client    ├────►│    NGINX    ├────►│  API (A/B)  │
└─────────────┘     │  Load       │     │  - Group A  │
                    │  Balancer   │     │  - Group B  │
                    └──────┬──────┘     └─────────────┘
                           │
                    ┌──────▼──────┐     ┌─────────────┐
                    │  Logging    │◄───►│ A/B Test    │
                    │  Service    │     │  Evaluator  │
                    └─────────────┘     └─────────────┘
```

## Components

### 1. NGINX Load Balancer
- Routes traffic between production (A) and shadow (B) API instances
- 50/50 traffic split by default (configurable in `nginx/nginx.conf`)
- Logs all routing decisions

### 2. API Instances
- **Production (Group A)**: Serves the current production model
- **Shadow (Group B)**: Serves the candidate model for testing
- Both instances log predictions to a shared volume

### 3. A/B Test Evaluator
- Continuously monitors and evaluates A/B test metrics
- Generates reports in the `reports/` directory
- Runs statistical tests to determine significant differences

## Setup

1. **Build and start the services**:
   ```bash
   docker-compose up --build -d
   ```

2. **Verify the services are running**:
   ```bash
   docker-compose ps
   ```

3. **Access the API**:
   - Main API: `http://localhost/api/v1/...`
   - A/B Test Summary: `http://localhost/api/v1/management/ab-test/summary`

## Testing the Setup

1. **Run the test script**:
   ```bash
   python scripts/test_ab_testing.py
   ```

2. **Check the A/B test summary**:
   ```bash
   curl http://localhost/api/v1/management/ab-test/summary | jq
   ```

3. **View the evaluation reports**:
   ```bash
   ls -l reports/
   ```

## Configuration

### NGINX Configuration
Edit `nginx/nginx.conf` to modify:
- Traffic split ratio
- Upstream servers
- Logging settings

### A/B Test Evaluator
Edit the `ab-test-evaluator` service in `docker-compose.yml` to adjust:
- Evaluation frequency
- Report generation settings
- Log file locations

## Monitoring

### Logs
- API logs: `docker-compose logs -f inference-api`
- NGINX logs: `docker-compose logs -f nginx`
- A/B Test logs: `tail -f logs/ab_test_logs/*.json`

### Metrics
- **Accuracy**: Percentage of correct predictions
- **Prediction Time**: Average time per prediction
- **Traffic Distribution**: Requests per group
- **Statistical Significance**: p-value for differences between groups

## Troubleshooting

### Common Issues
1. **NGINX not starting**:
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
