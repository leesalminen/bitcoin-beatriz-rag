-- Enable the pgvector extension (if not already enabled)
create extension if not exists vector with schema public;

-- Create the user table
CREATE TABLE "user" (
    id SERIAL PRIMARY KEY,
    username VARCHAR(80) UNIQUE NOT NULL,
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