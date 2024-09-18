# AI Chat Application

This project is an AI-powered chat application that allows users to interact with different AI personalities.

## Features

- User registration and login
- Create and manage AI personality pairs
- Chat interface with AI personalities

## Prerequisites

- Docker
- Docker Compose

## Setup

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/ai-chat-app.git
   cd ai-chat-app
   ```

2. Copy the `.env.example` file to `.env` and fill in your API keys and other configuration:
   ```
   cp .env.example .env
   ```

3. Build and run the Docker containers:
   ```
   docker-compose up --build
   ```

4. Access the application at `http://localhost:8000`

## Usage

1. Register a new account or log in
2. Create AI personality pairs in the "Manage Pairs" section
3. Start chatting with your created AI personalities

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

[MIT](https://choosealicense.com/licenses/mit/)
