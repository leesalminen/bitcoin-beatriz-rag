-- Add sender_id column to conversation table for human interaction detection
-- This allows us to track different users in group conversations

ALTER TABLE conversation 
ADD COLUMN sender_id VARCHAR(50);

-- Create index for better performance on sender_id queries
CREATE INDEX idx_conversation_sender_id ON conversation(sender_id);

-- Create composite index for phone_number + timestamp queries used in human detection