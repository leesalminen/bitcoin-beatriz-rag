-- Migration for system_prompts table

CREATE TABLE system_prompts (
    id SERIAL PRIMARY KEY,
    prompt_type VARCHAR(50) UNIQUE NOT NULL,
    content TEXT NOT NULL,
    last_modified TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Function to update last_modified timestamp on row update
CREATE OR REPLACE FUNCTION update_last_modified_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.last_modified = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Trigger to update last_modified on system_prompts table update
CREATE TRIGGER update_system_prompts_last_modified
BEFORE UPDATE ON system_prompts
FOR EACH ROW
EXECUTE FUNCTION update_last_modified_column();
