-- Migration: Update embeddings from 384 to 768 dimensions (Gemini Embedding 001)
-- This migration updates the vector column and index for the new embedding model

-- Step 1: Drop the existing index (required before altering the column)
DROP INDEX IF EXISTS prompt_completion_embedding_idx;

-- Step 2: Alter the embedding column to use 768 dimensions
-- Note: This will make existing embeddings incompatible, they must be recomputed
ALTER TABLE prompt_completion
    ALTER COLUMN embedding TYPE VECTOR(768);

-- Step 3: Clear existing embeddings (they're incompatible with the new dimension)
-- The embeddings need to be recomputed using the new Gemini model
UPDATE prompt_completion SET embedding = NULL;

-- Step 4: Recreate the index with the new dimension
-- Using IVFFlat for approximate nearest neighbor search
-- Note: The index will only be effective after embeddings are recomputed
CREATE INDEX prompt_completion_embedding_idx
    ON prompt_completion
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

-- Add a comment to document the change
COMMENT ON COLUMN prompt_completion.embedding IS
    'Embedding vector (768 dimensions) from Gemini Embedding 001 via OpenRouter API';
