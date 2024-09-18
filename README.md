# AI Chat RAG Application

This project is an AI-powered chat application that uses a Retrieval-Augmented Generation (RAG) system to provide context-aware responses. It leverages RunPod.io for scalable GPU-accelerated inference and PostgreSQL with pg_vector for efficient vector storage and similarity search.

## Features

- User registration and login
- Create and manage document pairs for RAG
- Chat interface with AI using RAG for enhanced responses
- Scalable GPU inference using RunPod.io
- Efficient vector storage and similarity search using PostgreSQL with pg_vector

## Prerequisites

- Docker
- Docker Compose
- RunPod.io account and API key

## Setup

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/ai-chat-rag-app.git
   cd ai-chat-rag-app
   ```

2. Copy the `.env.example` file to `.env` and fill in your API keys and other configuration:
   ```
   cp .env.example .env
   ```
   Make sure to add your RunPod.io API key and PostgreSQL credentials to the `.env` file.

3. Build and run the Docker containers:
   ```
   docker-compose up --build
   ```

4. Access the application at `http://localhost:8000`

## Usage

1. Register a new account or log in
2. Upload and manage document pairs in the "Manage Pairs" section
3. Start chatting with the AI, which will use the RAG system to provide context-aware responses
4. The application will use RunPod.io for GPU-accelerated inference and PostgreSQL for efficient vector operations

## RunPod.io Integration

This application uses RunPod.io to handle GPU-accelerated inference for the AI model. RunPod.io provides on-demand GPU resources, allowing the application to scale based on demand and provide faster response times.

To configure RunPod.io:
1. Sign up for a RunPod.io account
2. Obtain your API key from the RunPod.io dashboard
3. Add the API key to your `.env` file

## PostgreSQL with pg_vector

The application uses PostgreSQL with the pg_vector extension for efficient vector storage and similarity search. This allows for fast retrieval of relevant document pairs during the RAG process.

Key benefits:
- Efficient storage of high-dimensional vectors
- Fast similarity search using cosine distance
- Scalable solution for large document collections

The Docker setup includes a PostgreSQL container with pg_vector pre-installed and configured.

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

[MIT](https://choosealicense.com/licenses/mit/)