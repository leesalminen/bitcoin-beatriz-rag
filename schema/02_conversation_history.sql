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