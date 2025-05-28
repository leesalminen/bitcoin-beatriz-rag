-- Migration script to add is_active and created_at columns to the user table

-- Add is_active column with default value true
ALTER TABLE "user" ADD COLUMN IF NOT EXISTS is_active BOOLEAN DEFAULT TRUE;

-- Add created_at column with default value of current timestamp
ALTER TABLE "user" ADD COLUMN IF NOT EXISTS created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP;

-- Update existing users to be active
UPDATE "user" SET is_active = TRUE WHERE is_active IS NULL;

-- Make the first user (lowest ID) an admin if no admins exist
UPDATE "user" 
SET is_admin = TRUE 
WHERE id = (SELECT MIN(id) FROM "user")
AND NOT EXISTS (SELECT 1 FROM "user" WHERE is_admin = TRUE);

-- Set created_at for existing users to a reasonable default if NULL
UPDATE "user" 
SET created_at = CURRENT_TIMESTAMP 
WHERE created_at IS NULL; 