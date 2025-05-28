# RAG Chat Application

A Flask application that demonstrates Retrieval‑Augmented Generation (RAG) using PostgreSQL and pg\_vector. The project provides a simple chat interface where responses can be enriched with contextual information retrieved from a document collection. LLM inference is performed through the [OpenRouter](https://openrouter.ai) API, which provides access to various language models.

## Features

- User registration and login system
- Upload, search and manage document pairs used for RAG
- Chat interface that retrieves relevant context before generating a response
- PostgreSQL with the `pg_vector` extension for vector search
- Support for any language model available on OpenRouter
- Optional WhatsApp integration via WA Sender
- Admin panel for content moderation and system prompt customization

## Requirements

The easiest way to run the project is with Docker and Docker Compose. If you wish to run it locally without containers you will need Python 3.9 or later and PostgreSQL with `pg_vector` installed.

## Getting Started

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/rag-chat-app.git
   cd rag-chat-app
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
| `OPENROUTER_API_KEY` | API key for OpenRouter. Get one at https://openrouter.ai/keys |
| `OPENROUTER_MODEL` | Model to use on OpenRouter (e.g., `anthropic/claude-3.5-sonnet`, `openai/gpt-4-turbo`, etc.). See https://openrouter.ai/models for available models. |
| `WA_SENDER_API_URL` | (Optional) WA Sender endpoint for WhatsApp integration. |
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

## Usage

### Adding Knowledge to the RAG System

1. Log in to the application
2. Navigate to "Add New Information"
3. Enter prompt-completion pairs that represent Q&A or knowledge snippets
4. Submit for admin approval (or auto-approve if you're an admin)

### Customizing System Prompts

Admins can customize the system prompts used for both the chat interface and WhatsApp bot:

1. Log in as an admin
2. Go to Admin Actions → Manage System Prompts
3. Edit the prompts to match your use case
4. Save changes

### Bulk Import

For large knowledge bases, use the bulk import feature:

1. Prepare a JSONL file with prompt-completion pairs
2. Log in as an admin
3. Go to Admin Actions → Upload JSONL
4. Select and upload your file

## Project Structure

- `app.py` – main Flask application
- `templates/` – HTML templates
- `schema/` – SQL files used to initialize the database
- `docker-compose.yml` – container configuration

## License

This project is released under the [MIT](https://choosealicense.com/licenses/mit/) license.
