-- Migration: Add conversation history table for WhatsApp chat tracking
-- Date: 2025-01-23
-- Description: Stores conversation history between users and Bitcoin Beatriz via WhatsApp

CREATE TABLE conversation (
    id SERIAL PRIMARY KEY,
    phone_number VARCHAR(20) NOT NULL,
    message TEXT NOT NULL,
    is_from_user BOOLEAN NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Index for efficient phone number lookups
CREATE INDEX idx_conversation_phone ON conversation(phone_number);

-- Index for timestamp-based queries
CREATE INDEX idx_conversation_timestamp ON conversation(timestamp);

-- Composite index for phone number + timestamp (most common query pattern)
CREATE INDEX idx_conversation_phone_timestamp ON conversation(phone_number, timestamp DESC);

-- Alter conversation table to add team_id
ALTER TABLE conversation ADD COLUMN team_id INTEGER;
-- Making team_id NOT NULL as per model definition in the task.
-- This implies existing data will need migration or a default value.
-- For now, focusing on schema change. A separate step would handle data migration.
ALTER TABLE conversation ALTER COLUMN team_id SET NOT NULL;
ALTER TABLE conversation ADD CONSTRAINT fk_conversation_team FOREIGN KEY (team_id) REFERENCES team (id) ON DELETE CASCADE;