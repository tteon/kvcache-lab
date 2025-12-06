# Chainlit Chat Interface

A standalone, Dockerized chat interface built with [Chainlit](https://github.com/Chainlit/chainlit) that connects to external AI servers or APIs.

## Features

- **Configurable Backend**: Easily switch between different API endpoints via the settings UI.
- **Dockerized**: specific `Dockerfile` and `docker-compose.yml` included for instant deployment.
- **Rich UI**: Built on Chainlit for a modern, responsive chat experience.
- **Adapter Pattern**: Extensible codebase to support various API protocols.

## Getting Started

### Prerequisites

- Docker & Docker Compose
- (Optional) Python 3.10+ for local development

### Running with Docker (Recommended)

1.  **Start the container**:
    ```bash
    docker-compose up -d --build
    ```

2.  **Access the app**:
    Open your browser and navigate to `http://localhost:8001`.

### Running Locally

1.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the app**:
    ```bash
    chainlit run app.py -w
    ```

## Configuration

Once the app is running, open the **Settings** menu (bottom left) to configure:

-   **Target Server URL**: The endpoint of your AI server (e.g., `http://192.168.1.5:8000/chat`).
-   **Connection Mode**:
    -   `Generic API`: Sends a standard JSON POST `{ "query": "message" }`.
    -   `Online Serving Platform`: Placeholder for specific platform integrations.
-   **API Key**: Optional bearer token for authentication.

## Project Structure

-   `app.py`: Main Chainlit application entry point.
-   `chat_service.py`: Handles API communication logic.
-   `Dockerfile` / `docker-compose.yml`: Container configuration.
