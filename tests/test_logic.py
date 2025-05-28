import unittest
from unittest.mock import patch, MagicMock
from app import (
    app, db, Team, User, TeamConfiguration, SystemPrompt,
    get_team_configuration, generate_ai_response, send_wa_message,
    get_system_prompt, DEFAULT_CHAT_UI_PROMPT, DEFAULT_WHATSAPP_PROMPT
)
from tests.base import BaseTestCase # Assuming BaseTestCase is in tests.base

# Mock the global genai object if it's configured at module level in app.py
# This is to prevent actual API calls during tests.
# If genai is configured per call, this might need adjustment.
# For now, assume genai.configure and genai.GenerativeModel are points of interaction.
mock_genai_configure = MagicMock()
mock_generative_model_instance = MagicMock()
mock_generative_model_instance.generate_content.return_value = MagicMock(text="Test AI Response")

mock_genai = MagicMock()
mock_genai.configure = mock_genai_configure
mock_genai.GenerativeModel.return_value = mock_generative_model_instance

class TestLogicFunctions(BaseTestCase):

    def setUp(self):
        super().setUp()
        # Create some base data
        self.team1 = Team(name="Logic Team 1")
        self.team2 = Team(name="Logic Team 2")
        db.session.add_all([self.team1, self.team2])
        db.session.commit()

        self.config1 = TeamConfiguration(
            team_id=self.team1.id,
            gemini_api_key="team1_gemini_key",
            wa_sender_api_url="https://team1.wa.sender/api",
            wa_sender_api_key="team1_wa_key",
            runpod_api_key="team1_runpod_key",
            runpod_endpoint="https://team1.runpod.endpoint",
            runpod_model="team1_runpod_model"
        )
        # Team 2 has no explicit config for some tests
        db.session.add(self.config1)
        db.session.commit()

    def tearDown(self):
        # Clean up logic-specific data
        TeamConfiguration.query.delete()
        SystemPrompt.query.delete()
        Team.query.delete() # Users are likely cascaded or handled in BaseTestCase if created
        db.session.commit()
        super().tearDown()


    def test_get_team_configuration(self):
        """Test retrieving team configuration."""
        retrieved_config = get_team_configuration(self.team1.id)
        self.assertIsNotNone(retrieved_config)
        self.assertEqual(retrieved_config.gemini_api_key, "team1_gemini_key")

        retrieved_config_non_existent = get_team_configuration(self.team2.id) # Team 2 has no config
        self.assertIsNone(retrieved_config_non_existent) # Expect None as per function's behavior

        retrieved_config_invalid_id = get_team_configuration(999) # Non-existent team
        self.assertIsNone(retrieved_config_invalid_id)
    
    @patch('app.genai', mock_genai) # Patch the genai module used in app.py
    @patch('app.get_relevant_context') # Mock RAG
    def test_generate_ai_response_team_config(self, mock_get_relevant_context):
        """Test generate_ai_response uses team-specific Gemini key."""
        mock_get_relevant_context.return_value = [] # Empty RAG context for simplicity
        
        # Seed a system prompt for team1 to ensure get_system_prompt works
        sp = SystemPrompt(team_id=self.team1.id, prompt_type='whatsapp', content='Team 1 WhatsApp Prompt')
        db.session.add(sp)
        db.session.commit()

        response = generate_ai_response("hello", "12345", team_id=self.team1.id)
        self.assertEqual(response, "Test AI Response")
        # Check if genai.configure was called with team1's key
        mock_genai.configure.assert_called_with(api_key="team1_gemini_key")

    @patch('app.genai', mock_genai)
    @patch('app.get_relevant_context')
    def test_generate_ai_response_missing_team_config_key(self, mock_get_relevant_context):
        """Test generate_ai_response when team config or Gemini key is missing."""
        mock_get_relevant_context.return_value = []
        
        # Team 2 has no TeamConfiguration row, so get_team_configuration will return None
        response = generate_ai_response("hello", "67890", team_id=self.team2.id)
        self.assertEqual(response, "Lo siento, el servicio de AI no está configurado para este equipo.")

        # Create config for team2 but without a gemini key
        config2_no_gemini = TeamConfiguration(team_id=self.team2.id)
        db.session.add(config2_no_gemini)
        db.session.commit()
        response = generate_ai_response("hello", "67890", team_id=self.team2.id)
        self.assertEqual(response, "Lo siento, el servicio de AI no está configurado para este equipo.")


    @patch('app.requests.post')
    def test_send_wa_message_team_config(self, mock_post):
        """Test send_wa_message uses team-specific WA Sender config."""
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        success = send_wa_message("12345", "Test message", team_id=self.team1.id)
        self.assertTrue(success)
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        self.assertEqual(args[0], "https://team1.wa.sender/api")
        self.assertEqual(kwargs['headers']['Authorization'], "Bearer team1_wa_key")

    @patch('app.requests.post')
    def test_send_wa_message_missing_team_config(self, mock_post):
        """Test send_wa_message when team config for WA Sender is missing."""
        success = send_wa_message("67890", "Test message", team_id=self.team2.id) # Team 2 has no config
        self.assertFalse(success)
        mock_post.assert_not_called()

        # Create config for team2 but without wa sender url/key
        config2_no_wa = TeamConfiguration(team_id=self.team2.id)
        db.session.add(config2_no_wa)
        db.session.commit()
        success = send_wa_message("67890", "Test message", team_id=self.team2.id)
        self.assertFalse(success)
        mock_post.assert_not_called()

    def test_get_system_prompt_team_scoping(self):
        """Test get_system_prompt retrieves team-specific and default prompts."""
        # Team 1 has specific 'chat_ui'
        prompt_t1_chat = SystemPrompt(team_id=self.team1.id, prompt_type='chat_ui', content='Team 1 Chat UI')
        # Team 1 also has specific 'whatsapp'
        prompt_t1_wa = SystemPrompt(team_id=self.team1.id, prompt_type='whatsapp', content='Team 1 WhatsApp')
        # Team 2 has specific 'chat_ui' only
        prompt_t2_chat = SystemPrompt(team_id=self.team2.id, prompt_type='chat_ui', content='Team 2 Chat UI')
        db.session.add_all([prompt_t1_chat, prompt_t1_wa, prompt_t2_chat])
        db.session.commit()

        # Test Team 1
        self.assertEqual(get_system_prompt('chat_ui', team_id=self.team1.id), 'Team 1 Chat UI')
        self.assertEqual(get_system_prompt('whatsapp', team_id=self.team1.id), 'Team 1 WhatsApp')
        
        # Test Team 2
        self.assertEqual(get_system_prompt('chat_ui', team_id=self.team2.id), 'Team 2 Chat UI')
        # Team 2 does not have 'whatsapp' prompt, should fallback to global default
        self.assertEqual(get_system_prompt('whatsapp', team_id=self.team2.id), DEFAULT_WHATSAPP_PROMPT)

        # Test non-existent prompt_type for a team, should fallback to global default for that type
        # (assuming 'non_existent_type' is not in DEFAULT_PROMPTS map handled by get_system_prompt)
        # Based on current get_system_prompt, it will log error and return "Error: Unknown system prompt type"
        self.assertIn("Error: Unknown system prompt type", get_system_prompt('non_existent_type', team_id=self.team1.id))

        # Test with no team_id (should use global logic)
        # This requires a global prompt 'chat_ui' to exist for the non-team-aware part of get_system_prompt to find it,
        # or it will use DEFAULT_CHAT_UI_PROMPT.
        # To make this testable without depending on global data, we can check if it tries to query without team_id.
        # For now, we'll assume the fallback to DEFAULT_CHAT_UI_PROMPT if no global one is found in DB.
        self.assertEqual(get_system_prompt('chat_ui'), DEFAULT_CHAT_UI_PROMPT)


if __name__ == '__main__':
    unittest.main()
