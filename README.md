# Bitcoin Beatriz RAG Chat

Bitcoin Beatriz is a Flask application that demonstrates Retrieval‑Augmented Generation (RAG) using PostgreSQL and pg\_vector. The project provides a simple chat interface where responses can be enriched with contextual information retrieved from a document collection. GPU‑accelerated inference is performed through the [RunPod](https://runpod.io) API.

## Features

- User registration and login
- Upload, search and manage document pairs used for RAG
- Chat interface that retrieves relevant context before generating a response
- PostgreSQL with the `pg_vector` extension for vector search
- Optional WhatsApp integration via WA Sender

A public demo is available at [https://chat-assist.bitcoinjungle.app](https://chat-assist.bitcoinjungle.app).

## Requirements

The easiest way to run the project is with Docker and Docker Compose. If you wish to run it locally without containers you will need Python 3.9 or later and PostgreSQL with `pg_vector` installed.

## Getting Started

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/bitcoin-beatriz-rag.git
   cd bitcoin-beatriz-rag
   ```
2. **Create a configuration file**
   ```bash
   cp .env.example .env
   ```
   Edit `.env` and provide values for the variables described in the next section.
3. **Build and start the services**
   ```bash
   docker-compose up --build
   ```
   The Flask application will be available at `http://localhost:8000` by default (or the port specified by `LOCAL_PORT`).

### Environment Variables

The following variables are read from the `.env` file and used by Docker Compose:

| Variable | Description |
| --- | --- |
| `DATABASE_URL` | SQLAlchemy connection string used by the Flask app. |
| `SECRET_KEY` | Secret key for Flask sessions. |
| `RUNPOD_API_KEY` | API key for RunPod inference. |
| `RUNPOD_ENDPOINT` | RunPod endpoint URL. |
| `RUNPOD_MODEL` | Name of the model to invoke on RunPod. |
| `WA_SENDER_API_URL` | (Optional) WA Sender endpoint. |
| `WA_SENDER_API_KEY` | (Optional) WA Sender API key. |
| `WA_SENDER_WEBHOOK_SECRET` | (Optional) Secret used to verify WA Sender webhooks. |
| `POSTGRES_USER` | Username for the PostgreSQL container. |
| `POSTGRES_PASSWORD` | Password for the PostgreSQL container. |
| `POSTGRES_DB` | Database name created inside PostgreSQL. |
| `LOCAL_PORT` | Port on the host to expose the Flask application (defaults to 8000). |
| `WORKERS` | Number of Gunicorn workers to run. |

### Running Without Docker (optional)

If PostgreSQL and the required Python packages are already installed locally you can run the application directly:

```bash
pip install -r requirements.txt
python app.py
```

The database schema SQL files in the `schema/` directory can be executed manually to create the required tables.

## Project Structure

- `app.py` – main Flask application
- `templates/` – HTML templates
- `schema/` – SQL files used to initialize the database
- `docker-compose.yml` – container configuration

## License

This project is released under the [MIT](https://choosealicense.com/licenses/mit/) license.
