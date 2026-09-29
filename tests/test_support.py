import os
os.environ['DATABASE_URL'] = 'sqlite://'
os.environ['SECRET_KEY'] = 'local-contract-test'
os.environ['OPENROUTER_API_KEY'] = 'fake-test-key'

from types import SimpleNamespace
from pathlib import Path
import tempfile
import json
import unittest
from unittest.mock import patch, Mock

import app as bot


class SupportContracts(unittest.TestCase):
    def test_history_excludes_current_and_auto_reply_before_limit(self):
        with bot.app.app_context():
            phone = 'test-history'
            old = bot.Conversation.add_message(phone, 'original question', True, phone)
            current = bot.Conversation.add_message(phone, 'follow-up', True, phone)
            bot.Conversation.add_message(phone, 'business greeting', False, 'auto_reply')
            rows = bot.Conversation.get_conversation_history(phone, limit=1, exclude_id=current.id)
            self.assertEqual([r.id for r in rows], [old.id])

    def test_semantic_ranking_deduplicates_without_vote_bonus(self):
        rows = [dict(id=1, prompt='Lost phone', completion='Manual review', upvotes=0, downvotes=0, similarity=.9),
                dict(id=2, prompt='Lost  phone', completion='Manual review', upvotes=2, downvotes=0, similarity=.89),
                dict(id=3, prompt='Codes', completion='Try WhatsApp', upvotes=0, downvotes=0, similarity=.8)]
        result = Mock()
        result.mappings.return_value.all.return_value = rows
        with bot.app.app_context(), patch.object(bot, 'compute_embedding_for_query', return_value=Mock(tolist=lambda: [0.1]*768)), patch.object(bot.db.session, 'execute', return_value=result) as execute:
            self.assertEqual([x['id'] for x in bot.get_relevant_context('lost number', top_k=2)], [1,3])
            statement, parameters = execute.call_args.args
            self.assertNotIn('upvotes', str(statement).split('ORDER BY')[1])
            self.assertEqual(parameters['candidate_count'], 16)

    def test_followup_context_is_in_retrieval_and_current_is_supplied_once(self):
        history = [SimpleNamespace(is_from_user=False, message='continuation'),
                   SimpleNamespace(is_from_user=False, message='MI CUENTA > SALDOS'),
                   SimpleNamespace(is_from_user=True, message='Where is my Bull fiat balance?')]
        response = Mock()
        response.json.return_value = {'model':'google/gemini-3.8-flash','choices':[{'message':{'content':'Use Secciones'},'finish_reason':'stop'}]}
        with patch.object(bot.Conversation,'get_conversation_history',return_value=history) as get_history, patch.object(bot,'get_relevant_context',return_value=[]) as retrieve, patch.object(bot,'get_system_prompt',return_value='Reference {rag_context}'), patch.object(bot.requests,'post',return_value=response) as post:
            answer = bot.generate_ai_response('That menu is missing','test',current_message_id=42)
        self.assertEqual(answer,'Use Secciones')
        get_history.assert_called_once_with('test',exclude_id=42)
        self.assertIn('Bull fiat balance',retrieve.call_args.args[0])
        payload = post.call_args.kwargs['json']
        self.assertEqual(len(payload['messages']),4)
        self.assertIn('MI CUENTA > SALDOS\ncontinuation',payload['messages'][2]['content'])
        self.assertEqual(payload['messages'][-1]['content'],[{'type':'text','text':'That menu is missing'}])

    def test_reactions_never_generate_or_store_a_turn(self):
        event={'message':{'id':'react-1','type':'reaction'},'conversation':{'phone_number':'test'}}
        with patch.object(bot.Conversation,'add_message') as add, patch.object(bot,'generate_ai_response') as generate:
            bot.process_kapso_message(event)
        add.assert_not_called();generate.assert_not_called()

    def test_image_cannot_override_human_pause(self):
        event={'message':{'id':'image-1','type':'image','kapso':{'media_url':'https://example.invalid/image'}},'conversation':{'phone_number':'test'}}
        with patch.object(bot.Conversation,'add_message',return_value=SimpleNamespace(id=42)), patch.object(bot,'should_respond_to_message',return_value=(False,'Human operator has responded recently - bot paused')), patch.object(bot,'generate_ai_response') as generate:
            bot.process_kapso_message(event)
        generate.assert_not_called()

    def test_content_release_is_repeatable_and_rollback_preserves_messages(self):
        import numpy as np
        from ops import support_release as release
        with bot.app.app_context(), tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            owner = bot.User(username='release-admin',password='test',is_admin=True,is_active=True)
            bot.db.session.add(owner);bot.db.session.flush()
            old = bot.PromptCompletion(prompt='Old transfer-code flow',completion='Requires code',
                                       user_id=owner.id,is_approved=True,embedding=[0.1]*768)
            bot.db.session.add(old);bot.db.session.commit()
            old_id = old.id
            (root/'retire.json').write_text(json.dumps([{'id':old_id,'sha256':release.fingerprint(old)}]))
            (root/'knowledge.jsonl').write_text(json.dumps({'prompt':'Current deposit flow','completion':'No code'}))
            (root/'prompt.txt').write_text('Approved policy {rag_context}')
            (root/'chat_ui_prompt.txt').write_text('Approved web policy {rag_context}')
            record = root/'record.json'
            original_prompts = {p.id:p.content for p in bot.SystemPrompt.query.all()}
            original_messages = bot.Conversation.query.count()
            with patch.object(release,'compute_embedding_for_document',return_value=np.ones(768)):
                release.prepare(root,record)
            release.apply(record)
            row_count = bot.PromptCompletion.query.count()
            release.apply(record)
            self.assertEqual(bot.PromptCompletion.query.count(),row_count)
            self.assertFalse(bot.db.session.get(bot.PromptCompletion,old_id).is_approved)
            release.rollback(record)
            self.assertTrue(bot.db.session.get(bot.PromptCompletion,old_id).is_approved)
            self.assertEqual({p.id:p.content for p in bot.SystemPrompt.query.all()},original_prompts)
            self.assertEqual(bot.Conversation.query.count(),original_messages)


if __name__ == '__main__':
    unittest.main()
