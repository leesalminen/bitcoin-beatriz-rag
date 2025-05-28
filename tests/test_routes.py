import unittest
from unittest.mock import patch
from app import app, db, User, Team
from tests.base import BaseTestCase
from werkzeug.security import check_password_hash

class TestAuthRoutes(BaseTestCase):

    def tearDown(self):
        """Override teardown to clean up users and teams created in tests."""
        # Order of deletion matters due to foreign key constraints
        users = User.query.all()
        for user in users:
            db.session.delete(user)
        
        teams = Team.query.all()
        for team in teams:
            # Configurations and prompts are deleted via cascade if set up correctly,
            # otherwise, they might need explicit deletion here.
            # For now, assuming cascade delete from Team or manual cleanup elsewhere if needed.
            db.session.delete(team)
        
        db.session.commit()
        super().tearDown() # Call parent's teardown

    @patch('app.seed_team_configuration')
    @patch('app.seed_initial_prompts')
    def test_register_creates_user_team_and_calls_seeding(
            self, 
            mock_seed_prompts, 
            mock_seed_config):
        """Test user registration creates a user, a team, links them, and calls seeding functions."""
        
        test_email = "new_owner@example.com"
        test_password = "secure_password"

        response = self.client.post('/register', data={
            'email': test_email,
            'password': test_password
        }, follow_redirects=True)

        self.assertEqual(response.status_code, 200) # Should redirect to login, then login page is 200
        
        # Verify user creation
        user = User.query.filter_by(email=test_email).first()
        self.assertIsNotNone(user)
        self.assertTrue(check_password_hash(user.password, test_password))
        self.assertTrue(user.is_owner)
        self.assertTrue(user.is_admin) # As per current registration logic
        self.assertIsNotNone(user.team_id)

        # Verify team creation
        team = Team.query.get(user.team_id)
        self.assertIsNotNone(team)
        self.assertEqual(team.name, f"{test_email}'s Team")
        
        # Verify user is part of the team
        self.assertIn(user, team.users)

        # Verify that seeding functions were called with the new team's ID
        mock_seed_prompts.assert_called_once_with(team.id)
        mock_seed_config.assert_called_once_with(team.id)

        # Clean up (specific to this test, though tearDown should also run)
        # db.session.delete(user)
        # db.session.delete(team)
        # db.session.commit()

    def test_register_existing_user(self):
        """Test registration with an existing email."""
        # First, create a user
        team = Team(name="Existing User Team")
        db.session.add(team)
        db.session.commit()
        
        existing_email = "existing@example.com"
        existing_user = User(email=existing_email, password="password123", team_id=team.id)
        db.session.add(existing_user)
        db.session.commit()

        response = self.client.post('/register', data={
            'email': existing_email,
            'password': 'new_password'
        }, follow_redirects=True)
        
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Email already exists', response.data)

        # Clean up
        # db.session.delete(existing_user)
        # db.session.delete(team)
        # db.session.commit()

    def test_login_logout(self):
        """Test user login and logout."""
        # Create a team and user first
        team = Team(name="Login Test Team")
        db.session.add(team)
        db.session.commit()

        email = "login_user@example.com"
        password = "login_password"
        user = User(email=email, password=password, team_id=team.id) # Password will be hashed by User model if it had a setter
        # For test purposes, we'll use generate_password_hash directly as in register route
        from werkzeug.security import generate_password_hash
        user.password = generate_password_hash(password)
        db.session.add(user)
        db.session.commit()

        # Test login
        response = self.client.post('/login', data={
            'email': email,
            'password': password
        }, follow_redirects=True)
        
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Dashboard', response.data) # Assuming Dashboard is on index page after login
        with self.client.session_transaction() as sess:
            self.assertEqual(sess['user_id'], user.id)
            self.assertEqual(sess['team_id'], team.id)

        # Test logout
        response = self.client.get('/logout', follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Login', response.data) # Should redirect to login page
        with self.client.session_transaction() as sess:
            self.assertNotIn('user_id', sess)
            self.assertNotIn('team_id', sess)
        
        # Clean up
        # db.session.delete(user)
        # db.session.delete(team)
        # db.session.commit()

    def test_manage_pairs_is_team_scoped(self):
        """Test that /manage_pairs only shows prompts for the logged-in user's team."""
        # Create Team A and User A
        team_a = Team(name="Team A for Scoping")
        db.session.add(team_a)
        db.session.commit()
        user_a_email = "user_a_scope@example.com"
        user_a_password = "password_a"
        user_a = User(email=user_a_email, password=generate_password_hash(user_a_password), team_id=team_a.id, is_admin=True) # Admin to see manage_pairs
        db.session.add(user_a)
        db.session.commit()

        # Create Team B
        team_b = Team(name="Team B for Scoping")
        db.session.add(team_b)
        db.session.commit()
        user_b_email = "user_b_scope@example.com" # Dummy user for team_b prompts
        user_b = User(email=user_b_email, password="password_b", team_id=team_b.id)
        db.session.add(user_b)
        db.session.commit()


        # Import PromptCompletion here to avoid circular import issues if any
        from app import PromptCompletion

        # Create prompts for Team A
        prompt_a1_text = "Prompt A1 Text"
        prompt_a1 = PromptCompletion(prompt=prompt_a1_text, completion="Completion A1", user_id=user_a.id, team_id=team_a.id, is_approved=True)
        prompt_a2_text = "Prompt A2 Text"
        prompt_a2 = PromptCompletion(prompt=prompt_a2_text, completion="Completion A2", user_id=user_a.id, team_id=team_a.id, is_approved=True)
        
        # Create prompt for Team B
        prompt_b1_text = "Prompt B1 Text - Should Not Be Visible"
        prompt_b1 = PromptCompletion(prompt=prompt_b1_text, completion="Completion B1", user_id=user_b.id, team_id=team_b.id, is_approved=True)
        
        db.session.add_all([prompt_a1, prompt_a2, prompt_b1])
        db.session.commit()

        # Log in as User A
        self.client.post('/login', data={
            'email': user_a_email,
            'password': user_a_password
        }, follow_redirects=True)

        # Access /manage_pairs
        response = self.client.get('/manage_pairs')
        self.assertEqual(response.status_code, 200)
        
        response_data_str = response.data.decode('utf-8')

        # Verify Team A's prompts are present
        self.assertIn(prompt_a1_text, response_data_str)
        self.assertIn(prompt_a2_text, response_data_str)
        
        # Verify Team B's prompt is NOT present
        self.assertNotIn(prompt_b1_text, response_data_str)

        # Logout and cleanup is handled by tearDown

if __name__ == '__main__':
    unittest.main()
