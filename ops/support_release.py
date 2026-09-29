"""Apply the reviewed support content; keep only the changed rows for rollback."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import app, db, PromptCompletion, SystemPrompt, User
from embeddings import compute_embedding_for_document


def fingerprint(row):
    return hashlib.sha256(json.dumps(
        [row.prompt, row.completion], ensure_ascii=False
    ).encode()).hexdigest()


def save(path, record):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2))
    temporary.chmod(0o600)
    temporary.replace(path)


def prepare(content, path):
    if path.exists():
        raise RuntimeError('Prepared record already exists; use that record rather than overwriting rollback data')
    retire = json.loads((content / 'retire.json').read_text())
    previous = []
    for item in retire:
        row = db.session.get(PromptCompletion, item['id'])
        if row is None or fingerprint(row) != item['sha256']:
            raise RuntimeError(f"Knowledge entry {item['id']} changed since review")
        previous.append(dict(item, was_approved=row.is_approved))
    prompts = [dict(id=p.id, content=p.content) for p in
               SystemPrompt.query.filter(SystemPrompt.prompt_type.in_(['whatsapp', 'chat_ui'])).all()]
    if len(prompts) != 2:
        raise RuntimeError('Both system prompt records are required')
    owner = User.query.filter_by(is_admin=True, is_active=True).order_by(User.id).first()
    if owner is None:
        raise RuntimeError('An active knowledge administrator is required')
    owner_id = owner.id
    db.session.rollback()
    entries = []
    for index, line in enumerate((content / 'knowledge.jsonl').read_text().splitlines(), 1):
        entry = json.loads(line)
        vector = compute_embedding_for_document(f"Question: {entry['prompt']} Answer: {entry['completion']}")
        if len(vector) != 768 or not np.isfinite(vector).all() or not np.linalg.norm(vector):
            raise RuntimeError(f'Invalid embedding at entry {index}')
        entries.append(dict(entry, embedding=vector.tolist()))
        print(f'Prepared embedding {index}', flush=True)
    prompt = (content / 'prompt.txt').read_text()
    prompt.format(rag_context='format check')
    chat_ui_prompt = (content / 'chat_ui_prompt.txt').read_text()
    chat_ui_prompt.format(rag_context='format check')
    save(path, dict(previous=previous, prompts=prompts, entries=entries,
                    prompt=prompt, chat_ui_prompt=chat_ui_prompt, owner_id=owner_id, applied=False))


def apply(path):
    record = json.loads(path.read_text())
    if record['applied']:
        print('Release already applied')
        return
    changes = []
    for item in record['previous']:
        row = db.session.get(PromptCompletion, item['id'], with_for_update=True)
        if row is None or fingerprint(row) != item['sha256'] or row.is_approved != item['was_approved']:
            raise RuntimeError(f"Knowledge entry {item['id']} changed since preparation")
        row.is_approved = False
    for entry in record['entries']:
        row = PromptCompletion.query.filter_by(prompt=entry['prompt'], completion=entry['completion']).first()
        old_approval = row.is_approved if row else False
        old_embedding = row.embedding.tolist() if row is not None and row.embedding is not None else None
        if row is None:
            row = PromptCompletion(user_id=record['owner_id'], **entry)
            db.session.add(row)
        row.is_approved = True
        row.embedding = entry['embedding']
        db.session.flush()
        changes.append(dict(id=row.id, was_approved=old_approval, previous_embedding=old_embedding,
                            sha256=fingerprint(row)))
    for previous in record['prompts']:
        prompt = db.session.get(SystemPrompt, previous['id'], with_for_update=True)
        if prompt is None or prompt.content != previous['content']:
            raise RuntimeError('A system prompt changed since preparation')
        prompt.content = record['prompt'] if prompt.prompt_type == 'whatsapp' else record['chat_ui_prompt']
    # Persist IDs before committing, so rollback remains possible if the process stops at commit.
    record['new_entries'] = changes
    save(path, record)
    db.session.commit()
    record['applied'] = True
    save(path, record)
    print(f"Applied: {len(record['previous'])} retired, {len(changes)} canonical entries, 2 prompts")


def rollback(path):
    record = json.loads(path.read_text())
    if not record['applied']:
        raise RuntimeError('Release is not recorded as applied; inspect any uncertain commit before rollback')
    for item in record['previous']:
        row = db.session.get(PromptCompletion, item['id'], with_for_update=True)
        if row is None or fingerprint(row) != item['sha256'] or row.is_approved:
            raise RuntimeError(f"Retired entry {item['id']} was subsequently edited")
        row.is_approved = item['was_approved']
    for item in record.get('new_entries', []):
        row = db.session.get(PromptCompletion, item['id'], with_for_update=True)
        if row is None or not row.is_approved or fingerprint(row) != item['sha256']:
            raise RuntimeError(f"Canonical entry {item['id']} was subsequently edited")
        row.is_approved = item['was_approved']
        row.embedding = item['previous_embedding']
    for previous in record['prompts']:
        prompt = db.session.get(SystemPrompt, previous['id'], with_for_update=True)
        expected = record['prompt'] if prompt.prompt_type == 'whatsapp' else record['chat_ui_prompt']
        if prompt.content != expected:
            raise RuntimeError('A system prompt was subsequently edited')
        prompt.content = previous['content']
    db.session.commit()
    record['applied'] = False
    save(path, record)
    print('Restored original approval states and prompts; conversation history untouched')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'apply', 'rollback'])
    parser.add_argument('--content', type=Path, default=Path(__file__).resolve().parents[1] / 'support/2026-09-29')
    parser.add_argument('--record', type=Path, required=True)
    args = parser.parse_args()
    os.umask(0o077)
    with app.app_context():
        try:
            prepare(args.content, args.record) if args.action == 'prepare' else globals()[args.action](args.record)
        except Exception:
            db.session.rollback()
            raise
