# Database Schema Migrations

## User Management Migration

The `add_user_active_status.sql` migration adds user management functionality to the application.

### Changes Made:

1. **New Columns Added to User Table:**
   - `is_active` (BOOLEAN) - Determines if a user can log in
   - `created_at` (TIMESTAMP) - Tracks when the user was created

2. **Default Behavior:**
   - First user registered is automatically admin and active
   - Subsequent users are inactive by default and require admin approval
   - Existing users (if any) will be set as active when migration runs

### Running the Migration:

Run this SQL script against your PostgreSQL database:

```bash
psql -U your_username -d your_database -f schema/add_user_active_status.sql
```

Or if using a database URL:

```bash
psql $DATABASE_URL -f schema/add_user_active_status.sql
```

### User Management Features:

After running this migration, admins can:
- View all users with their status and creation date
- Activate/deactivate user accounts
- Grant/revoke admin privileges
- Delete user accounts

### Security Notes:
- At least one admin must remain in the system
- Users cannot deactivate, remove admin privileges from, or delete their own account
- Inactive users cannot log in until activated by an admin 