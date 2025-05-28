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

-- Alter system_prompts table for team_id integration
ALTER TABLE system_prompts ADD COLUMN team_id INTEGER;

-- Making team_id NOT NULL as per model definition in the task.
-- This implies existing data will need migration or a default value.
-- For now, focusing on schema change. A separate step would handle data migration.
-- It's generally safer to add the column as nullable, migrate data, then set to NOT NULL.
-- However, following task instructions to make it NOT NULL directly in schema definition.
-- This will FAIL if there's existing data without a team_id.
-- For a fresh setup, this is fine.
ALTER TABLE system_prompts ALTER COLUMN team_id SET NOT NULL;

ALTER TABLE system_prompts ADD CONSTRAINT fk_system_prompts_team FOREIGN KEY (team_id) REFERENCES team (id) ON DELETE CASCADE;

-- Drop the old unique constraint on prompt_type
-- Assuming the constraint name is 'system_prompts_prompt_type_key'.
-- Use IF EXISTS to avoid error if the constraint name is different or doesn't exist.
ALTER TABLE system_prompts DROP CONSTRAINT IF EXISTS system_prompts_prompt_type_key;

-- Add a new unique constraint for (team_id, prompt_type)
ALTER TABLE system_prompts ADD CONSTRAINT uq_team_prompt_type UNIQUE (team_id, prompt_type);
