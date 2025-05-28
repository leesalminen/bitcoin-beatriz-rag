import unittest
from app import db, Team, User, TeamConfiguration, SystemPrompt
from tests.base import BaseTestCase
from sqlalchemy.exc import IntegrityError

class TestModelCreation(BaseTestCase):

    def test_create_team(self):
        """Test creating a Team instance."""
        team_name = "Test Team Alpha"
        team = Team(name=team_name)
        db.session.add(team)
        db.session.commit()
        
        retrieved_team = Team.query.filter_by(name=team_name).first()
        self.assertIsNotNone(retrieved_team)
        self.assertEqual(retrieved_team.name, team_name)
        # Clean up
        db.session.delete(retrieved_team)
        db.session.commit()

    def test_create_user(self):
        """Test creating a User instance linked to a Team."""
        team = Team(name="User Team")
        db.session.add(team)
        db.session.commit()

        user_email = "test_user@example.com"
        user = User(email=user_email, password="password123", team_id=team.id, is_owner=False)
        db.session.add(user)
        db.session.commit()

        retrieved_user = User.query.filter_by(email=user_email).first()
        self.assertIsNotNone(retrieved_user)
        self.assertEqual(retrieved_user.team_id, team.id)
        self.assertEqual(retrieved_user.team.name, "User Team")
        
        # Clean up
        db.session.delete(retrieved_user)
        db.session.delete(team)
        db.session.commit()

    def test_create_team_configuration(self):
        """Test creating a TeamConfiguration instance linked to a Team."""
        team = Team(name="Config Team")
        db.session.add(team)
        db.session.commit()

        config = TeamConfiguration(
            team_id=team.id,
            gemini_api_key="test_gemini_key"
        )
        db.session.add(config)
        db.session.commit()

        retrieved_config = TeamConfiguration.query.filter_by(team_id=team.id).first()
        self.assertIsNotNone(retrieved_config)
        self.assertEqual(retrieved_config.gemini_api_key, "test_gemini_key")
        self.assertEqual(retrieved_config.team.name, "Config Team")

        # Clean up
        db.session.delete(retrieved_config)
        db.session.delete(team)
        db.session.commit()

    def test_create_system_prompt(self):
        """Test creating a SystemPrompt instance linked to a Team."""
        team = Team(name="Prompt Team")
        db.session.add(team)
        db.session.commit()

        prompt_type = "test_chat"
        content = "This is a test system prompt."
        sys_prompt = SystemPrompt(
            team_id=team.id,
            prompt_type=prompt_type,
            content=content
        )
        db.session.add(sys_prompt)
        db.session.commit()

        retrieved_prompt = SystemPrompt.query.filter_by(team_id=team.id, prompt_type=prompt_type).first()
        self.assertIsNotNone(retrieved_prompt)
        self.assertEqual(retrieved_prompt.content, content)
        self.assertEqual(retrieved_prompt.team.name, "Prompt Team")

        # Clean up
        db.session.delete(retrieved_prompt)
        db.session.delete(team)
        db.session.commit()

    def test_system_prompt_unique_constraint(self):
        """Test unique constraint (team_id, prompt_type) for SystemPrompt."""
        team = Team(name="Constraint Test Team")
        db.session.add(team)
        db.session.commit()

        prompt1 = SystemPrompt(team_id=team.id, prompt_type="unique_prompt", content="Content 1")
        db.session.add(prompt1)
        db.session.commit()

        prompt2_same_type = SystemPrompt(team_id=team.id, prompt_type="unique_prompt", content="Content 2")
        db.session.add(prompt2_same_type)
        
        with self.assertRaises(IntegrityError):
            db.session.commit()
        
        db.session.rollback() # Rollback the failed transaction

        # Test with a different team, same prompt_type (should be allowed)
        team2 = Team(name="Constraint Test Team 2")
        db.session.add(team2)
        db.session.commit()
        
        prompt3_different_team = SystemPrompt(team_id=team2.id, prompt_type="unique_prompt", content="Content 3")
        db.session.add(prompt3_different_team)
        try:
            db.session.commit() # This should succeed
        except IntegrityError:
            self.fail("Should be able to add same prompt_type for a different team.")


        # Clean up
        # Need to query them again as IDs might have changed after rollback or new commits
        p1 = SystemPrompt.query.filter_by(team_id=team.id, prompt_type="unique_prompt").first()
        if p1: db.session.delete(p1)
        
        p3 = SystemPrompt.query.filter_by(team_id=team2.id, prompt_type="unique_prompt").first()
        if p3: db.session.delete(p3)
        
        db.session.delete(team)
        db.session.delete(team2)
        db.session.commit()

if __name__ == '__main__':
    unittest.main()
