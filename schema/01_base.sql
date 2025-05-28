-- Enable the pgvector extension (if not already enabled)
create extension if not exists vector with schema public;

-- Create the team table
CREATE TABLE team (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL
);

-- Create the user table
CREATE TABLE "user" (
    id SERIAL PRIMARY KEY,
    email VARCHAR(120) UNIQUE NOT NULL, -- Renamed from username, type changed to VARCHAR(120)
    password VARCHAR(255) NOT NULL
);

-- Create the prompt_completion table
CREATE TABLE prompt_completion (
    id SERIAL PRIMARY KEY,
    prompt VARCHAR(500) NOT NULL,
    completion VARCHAR(1000) NOT NULL,
    user_id INTEGER NOT NULL,
    upvotes INTEGER DEFAULT 0,
    downvotes INTEGER DEFAULT 0,
    embedding VECTOR(384),  -- Assuming 384-dimensional embeddings from 'all-MiniLM-L6-v2'
    FOREIGN KEY (user_id) REFERENCES "user" (id)
);

-- Create the vote table
CREATE TABLE vote (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL,
    prompt_id INTEGER NOT NULL,
    vote_type VARCHAR(10) NOT NULL,
    FOREIGN KEY (user_id) REFERENCES "user" (id) ON DELETE CASCADE,
    FOREIGN KEY (prompt_id) REFERENCES prompt_completion (id) ON DELETE CASCADE
);

-- Create an index on the embedding column for faster similarity searches
CREATE INDEX prompt_completion_embedding_idx ON prompt_completion USING ivfflat (embedding vector_cosine_ops);

-- Add an 'is_admin' column to the user table
ALTER TABLE "user" ADD COLUMN is_admin BOOLEAN DEFAULT FALSE;

-- Add an 'is_approved' column to the prompt_completion table
ALTER TABLE prompt_completion ADD COLUMN is_approved BOOLEAN DEFAULT FALSE;
ALTER TABLE prompt_completion 
ALTER COLUMN prompt TYPE TEXT,
ALTER COLUMN completion TYPE TEXT;

-- Add team_id and is_owner to user table
ALTER TABLE "user" ADD COLUMN team_id INTEGER;
ALTER TABLE "user" ADD CONSTRAINT fk_user_team FOREIGN KEY (team_id) REFERENCES team (id);
ALTER TABLE "user" ADD COLUMN is_owner BOOLEAN DEFAULT FALSE;

-- Create TeamConfiguration table
CREATE TABLE team_configuration (
    id SERIAL PRIMARY KEY,
    team_id INTEGER NOT NULL UNIQUE,
    runpod_api_key TEXT,
    runpod_endpoint VARCHAR(255),
    runpod_model VARCHAR(100),
    gemini_api_key TEXT,
    wa_sender_api_url VARCHAR(255),
    wa_sender_api_key TEXT,
    wa_sender_webhook_secret TEXT,
    FOREIGN KEY (team_id) REFERENCES team (id) ON DELETE CASCADE
);

-- Alter prompt_completion table
ALTER TABLE prompt_completion ADD COLUMN team_id INTEGER;
-- Making team_id NOT NULL as per model definition in the task.
-- This implies existing data will need migration or a default value.
-- For now, focusing on schema change. A separate step would handle data migration.
ALTER TABLE prompt_completion ALTER COLUMN team_id SET NOT NULL;
ALTER TABLE prompt_completion ADD CONSTRAINT fk_prompt_completion_team FOREIGN KEY (team_id) REFERENCES team (id) ON DELETE CASCADE;