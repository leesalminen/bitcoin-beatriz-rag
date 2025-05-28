import os
from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify, Response, stream_with_context
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from sentence_transformers import SentenceTransformer
import numpy as np
from sqlalchemy import desc
from functools import wraps
from pgvector.sqlalchemy import Vector
from sqlalchemy import select
from sqlalchemy.sql.expression import func
from sqlalchemy.exc import SQLAlchemyError, IntegrityError
import traceback
from sqlalchemy import cast
from sqlalchemy import text
import json
from werkzeug.utils import secure_filename
import requests
import logging
from dotenv import load_dotenv
import itertools
import hmac
import hashlib

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add this to your imports
from typing import List, Dict

# OpenRouter API configuration
OPENROUTER_API_KEY = os.environ.get('OPENROUTER_API_KEY')
OPENROUTER_MODEL = os.environ.get('OPENROUTER_MODEL', 'google/gemini-2.5-flash-preview-05-20')  # Default to Gemini 2.0 Flash
OPENROUTER_API_URL = 'https://openrouter.ai/api/v1/chat/completions'

# WA Sender API configuration
WA_SENDER_API_URL = os.environ.get('WA_SENDER_API_URL')
WA_SENDER_API_KEY = os.environ.get('WA_SENDER_API_KEY')
WA_SENDER_WEBHOOK_SECRET = os.environ.get('WA_SENDER_WEBHOOK_SECRET')

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL')
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY')
app.config['ALLOWED_EXTENSIONS'] = {'jsonl'}

db = SQLAlchemy(app)

class SystemPrompt(db.Model):
    __tablename__ = 'system_prompts' # Explicitly set table name
    id = db.Column(db.Integer, primary_key=True)
    prompt_type = db.Column(db.String(50), unique=True, nullable=False)
    content = db.Column(db.Text, nullable=False)
    last_modified = db.Column(db.TIMESTAMP, server_default=db.func.now(), onupdate=db.func.now())

# Initialize the sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# --- Default System Prompts ---
DEFAULT_CHAT_UI_PROMPT = """
You are a helpful AI assistant with access to a knowledge base. Your role is to provide accurate, informative, and helpful responses based on the context provided and your general knowledge.

**Core Principles:**
- **Accuracy First:** Provide factual, well-researched information
- **Context-Aware:** Use the provided context to enhance your responses
- **Clear Communication:** Express ideas clearly and concisely
- **User-Focused:** Tailor responses to the user's level of understanding

**Capabilities:**
- Answer questions based on the provided context
- Explain complex concepts in simple terms
- Provide balanced perspectives on topics
- Acknowledge limitations when information is unclear or unavailable

**Communication Style:**
- Professional yet approachable
- Structured and organized responses
- Use examples when helpful
- Adapt language complexity to match the user's needs

**Important Guidelines:**
- If the context doesn't contain relevant information, acknowledge this and provide the best answer from general knowledge
- Be transparent about uncertainty
- Correct any misconceptions respectfully
- Encourage follow-up questions for clarity

**Context Information:**
Below is relevant context that may help answer the user's question:

{rag_context}
"""

DEFAULT_WHATSAPP_PROMPT = """
You are a helpful WhatsApp AI assistant with access to a knowledge base. Your role is to provide quick, accurate, and helpful responses suitable for mobile messaging.

**CRITICAL RULE: Always respond in the SAME LANGUAGE as the user's message.**

**Core Objective:** Provide concise, clear, and helpful information in a WhatsApp-appropriate format.

**Key Principles:**
- **Brevity:** Keep responses short and to the point (1-3 short paragraphs max)
- **Mobile-Friendly:** Format for easy reading on small screens
- **Context-Aware:** Use provided context to enhance responses
- **Language Matching:** Always respond in the user's language

**Communication Guidelines:**

1. **BE CONCISE:**
   - Typical WhatsApp message length
   - Get to the point quickly
   - Avoid unnecessary elaboration

2. **HANDLING UNCLEAR MESSAGES:**
   - Try to infer intent from context
   - Ask clarifying questions when needed
   - Keep clarifications brief

3. **FORMATTING:**
   - Use short paragraphs
   - Strategic line breaks for readability
   - Minimal use of special formatting
   - Emojis sparingly, if appropriate

4. **TONE:**
   - Friendly and approachable
   - Professional yet conversational
   - Helpful and supportive

**Response Strategy:**
- Answer the direct question first
- Add relevant context only if helpful
- Suggest follow-up topics only when appropriate
- Keep technical jargon to a minimum

**Context Information:**
Below is relevant context that may help answer the user's question:

{rag_context}
"""

# --- System Prompt Function ---
def get_system_prompt(prompt_type: str) -> str:
    prompt_entry = SystemPrompt.query.filter_by(prompt_type=prompt_type).first()
    if prompt_entry:
        return prompt_entry.content
    else:
        logger.warning(f"System prompt '{prompt_type}' not found in database. Using default.")
        if prompt_type == 'chat_ui':
            return DEFAULT_CHAT_UI_PROMPT
        elif prompt_type == 'whatsapp':
            return DEFAULT_WHATSAPP_PROMPT
        else: # Should not happen if called correctly
            logger.error(f"Unknown prompt type requested: {prompt_type}")
            # Return a generic error prompt or the chat_ui one as a last resort.
            return "Error: Unknown system prompt type. Please check configuration."

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)
    is_admin = db.Column(db.Boolean, default=False)
    is_active = db.Column(db.Boolean, default=True)  # New field for active status
    created_at = db.Column(db.DateTime, default=db.func.current_timestamp())  # Track when user was created

class PromptCompletion(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    prompt = db.Column(db.Text, nullable=False)  # Changed from String to Text
    completion = db.Column(db.Text, nullable=False)  # Changed from String to Text
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    upvotes = db.Column(db.Integer, default=0)
    downvotes = db.Column(db.Integer, default=0)
    embedding = db.Column(Vector(384))
    is_approved = db.Column(db.Boolean, default=False)
    votes = db.relationship('Vote', backref='prompt_completion', cascade='all, delete-orphan')

class Vote(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    prompt_id = db.Column(db.Integer, db.ForeignKey('prompt_completion.id'), nullable=False)
    vote_type = db.Column(db.String(10), nullable=False)  # 'upvote' or 'downvote'

class Conversation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    phone_number = db.Column(db.String(20), nullable=False, index=True)
    message = db.Column(db.Text, nullable=False)
    is_from_user = db.Column(db.Boolean, nullable=False)  # True if from user, False if from bot
    timestamp = db.Column(db.DateTime, default=db.func.current_timestamp())
    sender_id = db.Column(db.String(50))  # WhatsApp sender ID to detect different users
    
    @classmethod
    def get_conversation_history(cls, phone_number: str, limit: int = 10):
        """Get recent conversation history for a phone number"""
        return cls.query.filter_by(phone_number=phone_number)\
                      .order_by(cls.timestamp.desc())\
                      .limit(limit)\
                      .all()
    
    @classmethod
    def add_message(cls, phone_number: str, message: str, is_from_user: bool, sender_id: str = None):
        """Add a message to conversation history"""
        conv = cls(phone_number=phone_number, message=message, is_from_user=is_from_user, sender_id=sender_id)
        db.session.add(conv)
        db.session.commit()
        return conv
    
    @classmethod
    def has_human_interaction_recently(cls, phone_number: str, minutes: int = 30) -> bool:
        """Check if there's been human interaction (multiple senders) in recent minutes"""
        from datetime import datetime, timedelta
        cutoff_time = datetime.utcnow() - timedelta(minutes=minutes)
        
        recent_messages = cls.query.filter(
            cls.phone_number == phone_number,
            cls.timestamp >= cutoff_time,
            cls.is_from_user == True
        ).all()
        
        # If we have messages from different sender_ids, there's human interaction
        sender_ids = set(msg.sender_id for msg in recent_messages if msg.sender_id)
        return len(sender_ids) > 1
    
    @classmethod
    def get_last_bot_response_time(cls, phone_number: str):
        """Get timestamp of last bot response"""
        last_bot_message = cls.query.filter(
            cls.phone_number == phone_number,
            cls.is_from_user == False
        ).order_by(cls.timestamp.desc()).first()
        
        return last_bot_message.timestamp if last_bot_message else None

def compute_embedding(text):
    return model.encode(text, convert_to_numpy=True)

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        user = User.query.get(session['user_id'])
        if not user or not user.is_admin:
            flash('Admin access required')
            return redirect(url_for('index'))
        return f(*args, **kwargs)
    return decorated_function

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        user = User.query.get(session['user_id'])
        if not user or not user.is_active:
            session.pop('user_id', None)  # Log out inactive users
            flash('Your account is not active. Please contact an administrator.')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/')
@login_required
def index():
    user = User.query.get(session['user_id'])
    return render_template('index.html', is_admin=user.is_admin)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            flash('Username already exists')
        else:
            # Check if this is the first user
            user_count = User.query.count()
            is_first_user = user_count == 0
            
            new_user = User(
                username=username, 
                password=generate_password_hash(password),
                is_admin=is_first_user,  # First user is admin
                is_active=is_first_user  # First user is active, others need approval
            )
            db.session.add(new_user)
            db.session.commit()
            
            if is_first_user:
                flash('Registration successful! You are the first user and have been granted admin privileges.')
            else:
                flash('Registration successful! Your account is pending admin approval.')
            return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            if not user.is_active:
                flash('Your account is pending admin approval. Please contact an administrator.')
                return render_template('login.html')
            session['user_id'] = user.id
            flash('Login successful')
            return redirect(url_for('index'))
        else:
            flash('Invalid username or password')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    return redirect(url_for('login'))

@app.route('/add', methods=['GET', 'POST'])
@login_required
def add_pair():
    if request.method == 'POST':
        prompt = request.form['prompt']
        completion = request.form['completion']
        combined_text = f"{prompt} {completion}"
        embedding = compute_embedding(combined_text)
        new_pair = PromptCompletion(
            prompt=prompt,
            completion=completion,
            user_id=session['user_id'],
            embedding=embedding,
            is_approved=False
        )
        db.session.add(new_pair)
        db.session.commit()
        flash('New information added successfully. Waiting for admin approval.')
        return redirect(url_for('manage_pairs'))  # Redirecting to manage_pairs after adding
    return render_template('add_pair.html')

@app.route('/pending_approvals')
@admin_required
def pending_approvals():
    page = request.args.get('page', 1, type=int)
    per_page = 10  # Number of items per page

    pending_pairs = PromptCompletion.query.filter_by(is_approved=False).\
        order_by(PromptCompletion.id.desc()).\
        paginate(page=page, per_page=per_page, error_out=False)

    return render_template('pending_approvals.html', pairs=pending_pairs)

@app.route('/approve/<int:id>')
@admin_required
def approve_pair(id):
    pair = PromptCompletion.query.get_or_404(id)
    pair.is_approved = True
    db.session.commit()
    flash('Information approved successfully')
    return redirect(url_for('pending_approvals'))

@app.route('/reject/<int:id>')
@admin_required
def reject_pair(id):
    pair = PromptCompletion.query.get_or_404(id)
    db.session.delete(pair)
    db.session.commit()
    flash('Information rejected and deleted')
    return redirect(url_for('pending_approvals'))

@app.route('/delete/<int:id>')
@login_required
def delete_pair(id):
    pair = PromptCompletion.query.get_or_404(id)
    if pair.user_id != session['user_id'] and not User.query.get(session['user_id']).is_admin:
        flash('Unauthorized')
        return redirect(url_for('index'))
    db.session.delete(pair)
    db.session.commit()
    flash('Information deleted successfully')
    return redirect(url_for('index'))

@app.route('/vote/<int:prompt_id>/<vote_type>')
@login_required
def vote(prompt_id, vote_type):
    user_id = session['user_id']
    prompt = PromptCompletion.query.get_or_404(prompt_id)
    existing_vote = Vote.query.filter_by(user_id=user_id, prompt_id=prompt_id).first()

    if existing_vote:
        if existing_vote.vote_type == vote_type:
            # Undo the vote
            db.session.delete(existing_vote)
            if vote_type == 'upvote':
                prompt.upvotes -= 1
            else:
                prompt.downvotes -= 1
        else:
            # Change the vote
            existing_vote.vote_type = vote_type
            if vote_type == 'upvote':
                prompt.upvotes += 1
                prompt.downvotes -= 1
            else:
                prompt.upvotes -= 1
                prompt.downvotes += 1
    else:
        # New vote
        new_vote = Vote(user_id=user_id, prompt_id=prompt_id, vote_type=vote_type)
        db.session.add(new_vote)
        if vote_type == 'upvote':
            prompt.upvotes += 1
        else:
            prompt.downvotes += 1

    db.session.commit()
    return jsonify({'success': True, 'upvotes': prompt.upvotes, 'downvotes': prompt.downvotes})

@app.route('/manage_pairs')
@login_required
def manage_pairs():
    user = User.query.get(session['user_id'])
    page = request.args.get('page', 1, type=int)
    per_page = 10  # Number of items per page

    pairs_query = PromptCompletion.query.filter_by(is_approved=True)

    # Add subquery to get user's vote for each prompt
    user_vote = db.session.query(Vote.prompt_id, Vote.vote_type).\
        filter(Vote.user_id == session['user_id']).\
        subquery()

    pairs = pairs_query.outerjoin(user_vote, PromptCompletion.id == user_vote.c.prompt_id).\
        add_columns(user_vote.c.vote_type.label('user_vote')).\
        order_by(desc(PromptCompletion.upvotes - PromptCompletion.downvotes)).\
        paginate(page=page, per_page=per_page, error_out=False)

    return render_template('manage_pairs.html', pairs=pairs, is_admin=user.is_admin)

@app.route('/recompute_embeddings')
@admin_required
def recompute_embeddings():
    pairs = PromptCompletion.query.all()
    for pair in pairs:
        combined_text = f"{pair.prompt} {pair.completion}"
        pair.embedding = compute_embedding(combined_text)
    db.session.commit()
    return jsonify({'success': True, 'message': 'Embeddings recomputed successfully'})

@app.route('/admin_actions')
@admin_required
def admin_actions():
    return render_template('admin_actions.html')

@app.route('/admin/system_prompts', methods=['GET'])
@admin_required
def manage_system_prompts_view():
    prompts = SystemPrompt.query.order_by(SystemPrompt.prompt_type).all()
    return render_template('manage_system_prompts.html', prompts=prompts)

@app.route('/admin/system_prompts/update', methods=['POST'])
@admin_required
def update_system_prompt_action():
    prompt_type = request.form.get('prompt_type')
    content = request.form.get('content')

    if not prompt_type or content is None: # content can be an empty string
        flash('Missing prompt_type or content.', 'error')
        return redirect(url_for('manage_system_prompts_view'))

    prompt_to_update = SystemPrompt.query.filter_by(prompt_type=prompt_type).first()
    if prompt_to_update:
        prompt_to_update.content = content
        # The last_modified timestamp will be updated automatically by the database trigger
        # or by SQLAlchemy's onupdate if that was configured on the model.
        # For SystemPrompt, it's db.Column(db.TIMESTAMP, server_default=db.func.now(), onupdate=db.func.now())
        # so SQLAlchemy should handle it.
        db.session.commit()
        flash(f"System prompt '{prompt_type}' updated successfully.", 'success')
    else:
        flash(f"System prompt type '{prompt_type}' not found.", 'error')
    
    return redirect(url_for('manage_system_prompts_view'))

@app.route('/api/search', methods=['POST'])
def search_vectors():
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({'error': 'No query provided'}), 400

        query = data['query']
        query_embedding = compute_embedding(query)

        # Convert numpy array to list and then to string
        query_vector_str = str(query_embedding.tolist())

        # Use text() to create a SQL expression with the vector as a string literal
        stmt = text(f"""
            SELECT id, prompt, completion, user_id, upvotes, downvotes, embedding::text, is_approved,
                   (1 - (embedding <=> '{query_vector_str}'::vector)) as cosine_similarity
            FROM prompt_completion
            WHERE is_approved = true
            ORDER BY 
                (1 - (embedding <=> '{query_vector_str}'::vector)) * 0.9 +
                (COALESCE(upvotes, 0) - COALESCE(downvotes, 0)) * 0.1 DESC
            LIMIT 5
        """)

        results = db.session.execute(stmt).fetchall()

        # Format the results
        formatted_results = []
        for result in results:
            formatted_results.append({
                'id': result.id,
                'prompt': result.prompt,
                'completion': result.completion,
                'similarity': result.cosine_similarity or 0,
                'net_votes': result.upvotes - result.downvotes,
                'upvotes': result.upvotes,
                'downvotes': result.downvotes
            })

        return jsonify(formatted_results)

    except SQLAlchemyError as e:
        db.session.rollback()
        app.logger.error(f"Database error: {str(e)}")
        return jsonify({'error': 'Database error occurred'}), 500
    except Exception as e:
        app.logger.error(f"Unexpected error: {str(e)}")
        app.logger.error(traceback.format_exc())
        return jsonify({'error': 'An unexpected error occurred'}), 500

# Add this new route to get the user's current vote
@app.route('/get_vote/<int:prompt_id>')
@login_required
def get_vote(prompt_id):
    user_id = session['user_id']
    vote = Vote.query.filter_by(user_id=user_id, prompt_id=prompt_id).first()
    return jsonify({'vote_type': vote.vote_type if vote else None})

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/upload', methods=['GET', 'POST'])
@admin_required
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            # Process the file in chunks
            chunk_size = 1000  # Number of lines to process at once
            total_processed = 0
            total_added = 0

            try:
                while True:
                    chunk = list(itertools.islice((line.strip() for line in file), chunk_size))
                    if not chunk:
                        break

                    new_pairs = []
                    for line in chunk:
                        try:
                            item = json.loads(line)
                            prompt = item.get('prompt')
                            completion = item.get('completion')
                            if prompt and completion:
                                combined_text = f"{prompt} {completion}"
                                embedding = compute_embedding(combined_text)
                                new_pair = PromptCompletion(
                                    prompt=prompt,
                                    completion=completion,
                                    user_id=session['user_id'],
                                    embedding=embedding,
                                    is_approved=True
                                )
                                new_pairs.append(new_pair)
                        except json.JSONDecodeError:
                            app.logger.warning(f'Invalid JSON in line: {line[:50]}...')
                            continue

                    if new_pairs:
                        try:
                            db.session.bulk_save_objects(new_pairs)
                            db.session.commit()
                            total_added += len(new_pairs)
                        except IntegrityError:
                            db.session.rollback()
                            app.logger.warning("Integrity error occurred during bulk insert. Some entries may be duplicates.")
                        except Exception as e:
                            db.session.rollback()
                            app.logger.error(f"Error during bulk insert: {str(e)}")

                    total_processed += len(chunk)
                    app.logger.info(f"Processed {total_processed} lines, added {total_added} entries")

                flash(f'File processed successfully. Added {total_added} out of {total_processed} entries.')
                return redirect(url_for('index'))

            except Exception as e:
                app.logger.error(f"Error processing file: {str(e)}")
                flash('An error occurred while processing the file.')
                return redirect(request.url)
    
    return render_template('upload.html')

def get_similar_vectors(query: str, top_k: int = 3) -> List[Dict]:
    query_embedding = compute_embedding(query)
    query_vector_str = str(query_embedding.tolist())

    stmt = text(f"""
        SELECT id, prompt, completion, user_id, upvotes, downvotes, embedding::text, is_approved,
               (1 - (embedding <=> '{query_vector_str}'::vector)) as cosine_similarity
        FROM prompt_completion
        WHERE is_approved = true
        ORDER BY 
            (1 - (embedding <=> '{query_vector_str}'::vector)) * 0.7 +
            (COALESCE(upvotes, 0) - COALESCE(downvotes, 0)) * 0.3 DESC
        LIMIT {top_k}
    """)

    results = db.session.execute(stmt).fetchall()
    return [{"prompt": r.prompt, "completion": r.completion} for r in results]

def get_relevant_context(query: str, top_k: int = 3) -> List[Dict]:
    return get_similar_vectors(query, top_k)

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        # Check if OpenRouter API key is configured
        if not OPENROUTER_API_KEY:
            return jsonify({'error': 'OpenRouter API key not configured. Please set OPENROUTER_API_KEY environment variable.'}), 500

        data = request.json
        if not data or 'messages' not in data:
            return jsonify({'error': 'Invalid request format'}), 400

        messages = data['messages']
        if not isinstance(messages, list) or len(messages) == 0:
            return jsonify({'error': 'Messages must be a non-empty list'}), 400

        last_user_message = next((m['content'] for m in reversed(messages) if m['role'] == 'user'), None)
        if not last_user_message:
            return jsonify({'error': 'No user message found'}), 400

        relevant_context = get_relevant_context(last_user_message)
        rag_context = "\n\n".join([f"Prompt: {ctx['prompt']}\nCompletion: {ctx['completion']}" for ctx in relevant_context])
        app.logger.info(f"RAG context for /api/chat: {rag_context[:200]}...") # Use info level for RAG, error was too much

        prompt_template = get_system_prompt('chat_ui')
        system_message_content = prompt_template.format(rag_context=rag_context)

        openrouter_messages = [{"role": "system", "content": system_message_content}] + messages

        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": request.headers.get('Referer', 'http://localhost:5000'),  # Optional but recommended
            "X-Title": "Bitcoin Beatriz RAG"  # Optional, helps OpenRouter understand your app
        }
        payload = {
            "model": OPENROUTER_MODEL,
            "messages": openrouter_messages,
            "stream": True  # Enable streaming
        }

        def generate():
            # Send "Thinking..." message
            yield 'data: {"id":"init","object":"chat.completion.chunk","created":1726594320,"model":"' + OPENROUTER_MODEL + '","choices":[{"index":0,"delta":{"role":"assistant", "content": "Thinking..."},"logprobs":null,"finish_reason":null}]}\n\n'
            thinking_cleared = False  # Flag to track if the message has been cleared            
            
            with requests.post(OPENROUTER_API_URL, json=payload, headers=headers, stream=True) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line:
                        if not thinking_cleared:
                            yield 'data: {"id":"init","object":"chat.completion.chunk","created":1726594320,"model":"' + OPENROUTER_MODEL + '","choices":[{"index":0,"delta":{"role":"assistant"},"logprobs":null,"finish_reason":null}]}\n\n'
                            thinking_cleared = True  # Set the flag to true
                        yield line.decode('utf-8') + "\n\n"

        return Response(stream_with_context(generate()), content_type='text/event-stream')

    except requests.RequestException as e:
        app.logger.error(f"Error calling OpenRouter API: {str(e)}")
        return jsonify({'error': 'Error communicating with AI service'}), 500
    except Exception as e:
        app.logger.error(f"Error in chat endpoint: {str(e)}")
        return jsonify({'error': 'An unexpected error occurred'}), 500

def send_wa_message(phone_number: str, message: str) -> bool:
    """Send a WhatsApp message using WA Sender API"""
    try:
        if not WA_SENDER_API_URL or not WA_SENDER_API_KEY:
            app.logger.error("WA Sender API configuration missing")
            return False
            
        headers = {
            'Authorization': f'Bearer {WA_SENDER_API_KEY}',
            'Content-Type': 'application/json'
        }
        
        payload = {
            'to': phone_number,
            'text': message
        }
        
        response = requests.post(f"{WA_SENDER_API_URL}/api/send-message", json=payload, headers=headers)
        response.raise_for_status()
        
        app.logger.info(f"Message sent successfully to {phone_number}")
        return True
        
    except requests.RequestException as e:
        app.logger.error(f"Error sending WA message: {str(e)}")
        return False
    except Exception as e:
        app.logger.error(f"Unexpected error sending WA message: {str(e)}")
        return False

def is_greeting_only(message: str) -> bool:
    """Check if message is just a greeting without a real question"""
    greetings = {
        'hi', 'hello', 'hey', 'hola', 'buenos dias', 'buenas tardes', 'buenas noches',
        'good morning', 'good afternoon', 'good evening', 'good night', 'que tal',
        'como estas', 'how are you', 'whats up', 'que pasa', 'saludos', 'holi'
    }
    
    # Clean and normalize the message
    clean_message = message.lower().strip()
    # Remove punctuation
    import string
    clean_message = clean_message.translate(str.maketrans('', '', string.punctuation))
    
    # Check if it's only greetings (allow some flexibility with extra words)
    words = clean_message.split()
    if len(words) <= 3:  # Short messages
        return all(word in greetings or word in ['que', 'como', 'are', 'you', 'estas'] for word in words)
    
    return False

def should_respond_to_message(phone_number: str, message: str, sender_id: str) -> tuple[bool, str]:
    """Determine if bot should respond to this message"""
    from datetime import datetime, timedelta
    
    # Check if there's been any human operator response in the last 60 minutes
    cutoff_time = datetime.utcnow() - timedelta(minutes=60)
    recent_operator_response = Conversation.query.filter(
        Conversation.phone_number == phone_number,
        Conversation.timestamp >= cutoff_time,
        Conversation.sender_id == 'human_operator'
    ).first()
    
    if recent_operator_response:
        return False, "Human operator has responded recently - bot paused"
    
    # Check if there's been any human response in the last 30 minutes
    # Look for messages that aren't from the bot
    cutoff_time_short = datetime.utcnow() - timedelta(minutes=30)
    recent_human_messages = Conversation.query.filter(
        Conversation.phone_number == phone_number,
        Conversation.timestamp >= cutoff_time_short,
        Conversation.is_from_user == True,
        Conversation.sender_id.notin_(['bot', 'human_operator'])  # Exclude bot and operator messages
    ).count()
    
    # If we have recent human messages and multiple different senders, pause bot
    if recent_human_messages > 0:
        recent_senders = db.session.query(Conversation.sender_id).filter(
            Conversation.phone_number == phone_number,
            Conversation.timestamp >= cutoff_time_short,
            Conversation.is_from_user == True,
            Conversation.sender_id.notin_(['bot', 'human_operator'])
        ).distinct().count()
        
        if recent_senders > 1:
            return False, "Multiple humans detected - bot paused"
    
    # Check if it's just a greeting
    if is_greeting_only(message):
        return False, "Greeting detected - waiting for real question"
    
    # Check if bot has responded recently (avoid spam)
    last_bot_time = Conversation.get_last_bot_response_time(phone_number)
    if last_bot_time:
        time_since_last = datetime.utcnow() - last_bot_time.replace(tzinfo=None)
        if time_since_last < timedelta(seconds=10):  # Wait at least 10 seconds between responses
            return False, "Bot cooling down period"
    
    return True, "OK to respond"

def generate_ai_response(user_message: str, phone_number: str) -> str:
    """Generate AI response using OpenRouter for WhatsApp with conversation history"""
    try:
        # Check if OpenRouter API key is configured
        if not OPENROUTER_API_KEY:
            app.logger.error("OpenRouter API key not configured")
            return "Lo siento, no pude procesar tu mensaje en este momento. Por favor intenta de nuevo más tarde."
            
        # Get conversation history
        conversation_history = Conversation.get_conversation_history(phone_number, limit=10)
        conversation_history.reverse()  # Oldest first for context
        
        relevant_context = get_relevant_context(user_message)
        rag_context = "\n\n".join([f"Prompt: {ctx['prompt']}\nCompletion: {ctx['completion']}" for ctx in relevant_context])
        app.logger.info(f"RAG context for {phone_number}: {rag_context[:200]}...")

        prompt_template = get_system_prompt('whatsapp')
        system_message_content = prompt_template.format(rag_context=rag_context)

        # Build messages array for OpenRouter
        messages = [{"role": "system", "content": system_message_content}]
        
        # Add conversation history
        for conv in conversation_history:
            if conv.is_from_user:
                messages.append({"role": "user", "content": conv.message})
            else:
                messages.append({"role": "assistant", "content": conv.message})
        
        # Add current user message
        messages.append({"role": "user", "content": user_message})
        
        # Prepare OpenRouter API request
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:5000",  # Required by OpenRouter
            "X-Title": "Bitcoin Beatriz WhatsApp Bot"
        }
        
        payload = {
            "model": OPENROUTER_MODEL,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 1000
        }
        
        response = requests.post(OPENROUTER_API_URL, json=payload, headers=headers)
        response.raise_for_status()
        
        response_data = response.json()
        
        if 'choices' in response_data and len(response_data['choices']) > 0:
            ai_response = response_data['choices'][0]['message']['content']
            app.logger.info(f"OpenRouter response for {phone_number}: {ai_response[:100]}...")
            return ai_response.strip()
        else:
            app.logger.warning(f"Unexpected response format from OpenRouter for {phone_number}")
            return "Lo siento, no pude generar una respuesta. Por favor intenta de nuevo."
        
    except requests.RequestException as e:
        app.logger.error(f"Error calling OpenRouter API: {str(e)}")
        if hasattr(e, 'response') and e.response is not None:
            app.logger.error(f"Response content: {e.response.text}")
        return "Lo siento, no pude procesar tu mensaje en este momento. Por favor intenta de nuevo más tarde."
    except Exception as e:
        app.logger.error(f"Error generating AI response with OpenRouter: {str(e)}")
        return "Lo siento, no pude procesar tu mensaje en este momento. Por favor intenta de nuevo más tarde."

def verify_webhook_signature(signature: str) -> bool:
    """Verify webhook signature from WA Sender API"""
    if not WA_SENDER_WEBHOOK_SECRET:
        app.logger.warning("Webhook secret not configured - skipping verification")
        return True
        
    try:
        # WasenderAPI uses direct secret comparison, not HMAC
        return hmac.compare_digest(signature, WA_SENDER_WEBHOOK_SECRET)
    except Exception as e:
        app.logger.error(f"Error verifying webhook signature: {str(e)}")
        return False

@app.route('/webhook', methods=['POST'])
def wa_webhook():
    """Handle incoming WhatsApp messages from WA Sender API"""
    try:
        # Verify webhook signature
        signature = request.headers.get('X-Webhook-Signature')
        if signature:
            if not verify_webhook_signature(signature):
                app.logger.warning("Invalid webhook signature")
                return jsonify({'error': 'Invalid signature'}), 401
        elif WA_SENDER_WEBHOOK_SECRET:
            app.logger.warning("No signature provided but secret is configured")
            return jsonify({'error': 'Signature required'}), 401
        
        data = request.get_json()
        app.logger.info(f"Received webhook data: {data}")
        
        if not data:
            app.logger.warning("No data received")
            return jsonify({'error': 'No data received'}), 400
            
        # Extract message details based on WasenderAPI format
        event_type = data.get('event')
        if event_type != 'messages.upsert':
            app.logger.info(f"Ignoring event type: {event_type}")
            return jsonify({'status': 'ignored'}), 200
            
        # Handle messages.upsert event
        message_data = data.get('data', {}).get('messages', {})
        if not message_data:
            app.logger.warning("No message data received")
            return jsonify({'status': 'no_messages'}), 200
            
        # Extract message details from the nested structure
        from_number = message_data.get('key', {}).get('remoteJid', '').replace('@s.whatsapp.net', '')
        
        # We'll let the intelligence logic handle whether to respond or not
        # Don't skip messages based on fromMe since human operators use the same account
            
        # Extract message text from various WhatsApp message formats
        message_obj = message_data.get('message', {})
        message_text = None
        
        # Check for regular conversation message
        if 'conversation' in message_obj:
            message_text = message_obj['conversation']
        
        # Check for extended text message
        elif 'extendedTextMessage' in message_obj:
            message_text = message_obj['extendedTextMessage'].get('text')
        
        # Check for ephemeral (disappearing) messages
        elif 'ephemeralMessage' in message_obj:
            ephemeral_msg = message_obj['ephemeralMessage'].get('message', {})
            # Try conversation first
            if 'conversation' in ephemeral_msg:
                message_text = ephemeral_msg['conversation']
            # Then try extended text
            elif 'extendedTextMessage' in ephemeral_msg:
                message_text = ephemeral_msg['extendedTextMessage'].get('text')
        
        if not from_number or not message_text:
            app.logger.warning(f"Missing required message data. from_number: {from_number}, message_text: {message_text}")
            app.logger.debug(f"Full message structure: {message_data}")
            return jsonify({'error': 'Invalid message format'}), 400
            
        app.logger.info(f"Processing message from {from_number}: {message_text}")
        
        # Extract sender ID for human interaction detection
        sender_id = message_data.get('key', {}).get('participant') or from_number
        
        # Check if this is a bot message (fromMe=True with AI-like patterns)
        is_from_me = message_data.get('key', {}).get('fromMe', False)
        
        # If it's from our account, check if it's a bot message
        if is_from_me:
            # Bot messages typically have these characteristics:
            is_likely_bot_message = (
                len(message_text) > 50 or  # Long responses typical of AI
                any(phrase in message_text.lower() for phrase in [
                    'bitcoin jungle', 'billetera', 'wallet', 'crypto', 'blockchain',
                    'descarga', 'install', 'dirección de bitcoin', 'transacción',
                    'seguridad', 'contraseña', 'copia de seguridad', 'backup'
                ]) or
                # Look for AI-like structured responses
                ('1.' in message_text and '2.' in message_text) or  # Numbered lists
                message_text.count('\n') > 2  # Multi-paragraph responses
            )
            
            if is_likely_bot_message:
                # This is a bot message - ignore it completely
                app.logger.info(f"Bot message detected and ignored for {from_number}")
                return jsonify({'status': 'bot_message_ignored'}), 200
                
            # If fromMe=True but doesn't look like bot message, treat as human operator
            Conversation.add_message(from_number, message_text, is_from_user=True, sender_id='human_operator')
            app.logger.info(f"Human operator response detected to {from_number}")
            return jsonify({'status': 'human_operator_response'}), 200
        
        # Store regular user message in conversation history
        Conversation.add_message(from_number, message_text, is_from_user=True, sender_id=sender_id)
        
        # Check if we should respond to this message
        should_respond, reason = should_respond_to_message(from_number, message_text, sender_id)
        
        if not should_respond:
            app.logger.info(f"Not responding to {from_number}: {reason}")
            return jsonify({'status': 'ignored', 'reason': reason}), 200
        
        # Generate AI response
        ai_response = generate_ai_response(message_text, from_number)
        
        # Store bot response in conversation history
        Conversation.add_message(from_number, ai_response, is_from_user=False, sender_id='bot')
        
        # Send response back via WA Sender API
        success = send_wa_message(from_number, ai_response)
        
        if success:
            app.logger.info(f"Successfully responded to {from_number}")
            return jsonify({'status': 'success', 'message': 'Response sent'}), 200
        else:
            app.logger.error(f"Failed to send response to {from_number}")
            return jsonify({'status': 'error', 'message': 'Failed to send response'}), 500
            
    except Exception as e:
        app.logger.error(f"Error processing webhook: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    db.session.rollback()
    return render_template('500.html'), 500

def seed_initial_prompts():
    """Seed initial system prompts if they don't exist"""
    initial_prompts = {
        'chat_ui': DEFAULT_CHAT_UI_PROMPT,
        'whatsapp': DEFAULT_WHATSAPP_PROMPT
    }
    
    for p_type, p_content in initial_prompts.items():
        existing_prompt = SystemPrompt.query.filter_by(prompt_type=p_type).first()
        if not existing_prompt:
            new_prompt = SystemPrompt(prompt_type=p_type, content=p_content)
            db.session.add(new_prompt)
            logger.info(f"Seeding system prompt: {p_type}")
    
    try:
        db.session.commit()
        logger.info("Initial system prompts seeded successfully")
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error seeding initial prompts: {str(e)}")

def init_db():
    """Initialize database and seed initial data"""
    try:
        db.create_all()
        logger.info("Database tables created successfully")
        seed_initial_prompts()
    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}")

# Initialize database when app starts (works for both direct run and WSGI)
with app.app_context():
    init_db()

@app.route('/admin/users')
@admin_required
def manage_users():
    page = request.args.get('page', 1, type=int)
    per_page = 10
    
    users = User.query.order_by(User.created_at.desc()).paginate(
        page=page, per_page=per_page, error_out=False
    )
    
    return render_template('manage_users.html', users=users)

@app.route('/admin/users/<int:user_id>/toggle_active', methods=['POST'])
@admin_required
def toggle_user_active(user_id):
    user = User.query.get_or_404(user_id)
    
    # Prevent deactivating yourself
    if user.id == session['user_id']:
        flash('You cannot deactivate your own account')
        return redirect(url_for('manage_users'))
    
    user.is_active = not user.is_active
    db.session.commit()
    
    status = 'activated' if user.is_active else 'deactivated'
    flash(f'User {user.username} has been {status}')
    return redirect(url_for('manage_users'))

@app.route('/admin/users/<int:user_id>/toggle_admin', methods=['POST'])
@admin_required
def toggle_user_admin(user_id):
    user = User.query.get_or_404(user_id)
    
    # Prevent removing your own admin status
    if user.id == session['user_id']:
        flash('You cannot remove your own admin privileges')
        return redirect(url_for('manage_users'))
    
    # Ensure at least one admin remains
    if user.is_admin and User.query.filter_by(is_admin=True).count() == 1:
        flash('Cannot remove admin privileges. At least one admin must remain.')
        return redirect(url_for('manage_users'))
    
    user.is_admin = not user.is_admin
    db.session.commit()
    
    status = 'granted admin privileges' if user.is_admin else 'removed admin privileges'
    flash(f'User {user.username} has been {status}')
    return redirect(url_for('manage_users'))

@app.route('/admin/users/<int:user_id>/delete', methods=['POST'])
@admin_required
def delete_user(user_id):
    user = User.query.get_or_404(user_id)
    
    # Prevent deleting yourself
    if user.id == session['user_id']:
        flash('You cannot delete your own account')
        return redirect(url_for('manage_users'))
    
    # Ensure at least one admin remains
    if user.is_admin and User.query.filter_by(is_admin=True).count() == 1:
        flash('Cannot delete the last admin user.')
        return redirect(url_for('manage_users'))
    
    username = user.username
    db.session.delete(user)
    db.session.commit()
    
    flash(f'User {username} has been deleted')
    return redirect(url_for('manage_users'))

if __name__ == '__main__':
    app.run(host='0.0.0.0')
