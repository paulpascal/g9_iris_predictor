# Iris Species Predictor

A simple machine learning project to predict the species of an Iris flower based on its features.

## Project Information

This project was developed by **Group9** as part of the **DevOps Master IA 1** course. The team members are:

- **Paul ALOGNON-ANANI**
- **Amal TANI NOUR**
- **Celestin PEHAN**

## Requirements

- Docker
- Docker Compose

## Setup

1. Clone the repository.
2. Build and run the Docker container:
    ```bash
    docker-compose up --build
    ```
3. Access the application at `http://localhost:5000`.

## Deployment

Deployment is handled via GitHub Actions. Push to the `main` branch triggers the deployment to AWS EC2.

## Testing

Run tests with:
```bash
pytest tests
```

## Cleanup

Artifacts older than 30 days are automatically cleaned up by a scheduled GitHub Action.
