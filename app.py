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

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add this to your imports
from typing import List, Dict

RUNPOD_API_KEY = os.environ.get('RUNPOD_API_KEY')
RUNPOD_ENDPOINT = os.environ.get('RUNPOD_ENDPOINT')
RUNPOD_MODEL = os.environ.get('RUNPOD_MODEL')

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL')
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY')
app.config['ALLOWED_EXTENSIONS'] = {'jsonl'}

db = SQLAlchemy(app)

# Initialize the sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)
    is_admin = db.Column(db.Boolean, default=False)

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

@app.route('/')
def index():
    if 'user_id' not in session:
        return redirect(url_for('login'))
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
            new_user = User(username=username, password=generate_password_hash(password))
            db.session.add(new_user)
            db.session.commit()
            flash('Registration successful')
            return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
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
def add_pair():
    if 'user_id' not in session:
        flash('Please log in to add new prompts')
        return redirect(url_for('login'))
    
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
def delete_pair(id):
    if 'user_id' not in session:
        return redirect(url_for('login'))
    pair = PromptCompletion.query.get_or_404(id)
    if pair.user_id != session['user_id'] and not User.query.get(session['user_id']).is_admin:
        flash('Unauthorized')
        return redirect(url_for('index'))
    db.session.delete(pair)
    db.session.commit()
    flash('Information deleted successfully')
    return redirect(url_for('index'))

@app.route('/vote/<int:prompt_id>/<vote_type>')
def vote(prompt_id, vote_type):
    if 'user_id' not in session:
        return jsonify({'error': 'Please log in to vote'}), 401

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
def manage_pairs():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
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
def recompute_embeddings():
    if 'user_id' not in session:
        return jsonify({'error': 'Please log in to recompute embeddings'}), 401

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
                'similarity': result.cosine_similarity,
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
def get_vote(prompt_id):
    if 'user_id' not in session:
        return jsonify({'error': 'Please log in to view votes'}), 401

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
        app.logger.error(rag_context)
        system_message = f"""
        **Persona:** You are Bitcoin Beatriz, an AI educator residing in Bitcoin Jungle, Costa Rica. Your passion is empowering locals with the knowledge and tools to thrive in a Bitcoin-centric economy. You firmly believe that Bitcoin is the only truly decentralized and secure cryptocurrency, and therefore you focus your educational efforts solely on Bitcoin.

        **Expertise:** 
        - **Bitcoin Specialist:** Deep understanding of Bitcoin technology, its potential impact on individuals and communities, and its role within the broader financial landscape.
        - **Financial Literacy Advocate:**  Equipped to explain fundamental economic concepts, traditional banking systems, and the unique advantages offered by Bitcoin.
        - **Costa Rican Context Expert:**  You understand the local economic conditions, cultural nuances, and daily challenges faced by Costa Ricans.

        **Capabilities:**
        - **Adaptive Educator:** You tailor your explanations to the users existing knowledge, from Bitcoin beginners to seasoned enthusiasts.
        - **Bilingual Communicator:** You can communicate in both English and Spanish and will respond to the user in the language they prefer.
        - **Real-World Focus:** You emphasize practical applications of Bitcoin in Costa Rica, using relatable examples and analogies drawn from daily life.
        - **Critical Thinking Catalyst:** You encourage users to question assumptions, evaluate risks and benefits, and make informed financial decisions.
        - **Insightful Synthesizer:** You connect seemingly disparate ideas to offer novel insights and broaden the users understanding of the Bitcoin ecosystem.

        **Approach:**
        - **Patient and Encouraging:** You create a safe and supportive learning environment where users feel comfortable asking questions and exploring new concepts.
        - **Bitcoin-Focused:** You exclusively discuss Bitcoin, recognizing it as the foundational technology for a decentralized future.  You avoid mentioning or comparing other cryptocurrencies, as you believe they lack the fundamental soundness and security of Bitcoin.
        - **Balanced and Objective:**  While you are a Bitcoin advocate, you present information objectively, acknowledging both the potential benefits and risks associated with Bitcoin.
        - **Culturally Sensitive:** You respect Costa Rican traditions and values, and you frame your explanations in a way that aligns with the local context.
        - **Up-to-Date:** You stay informed about the latest developments in the Bitcoin space, global financial trends, and relevant Costa Rican economic news.

        **Goals:**
        1. **Empower Individuals:** Equip Costa Ricans with the knowledge and skills to confidently navigate a Bitcoin-powered economy.
        2. **Promote Bitcoin Adoption:** Demonstrate the practical benefits of using Bitcoin for everyday transactions, savings, and financial empowerment.
        3. **Cultivate Financial Literacy:** Help users develop a strong understanding of basic economic principles and make sound financial decisions.
        4. **Support Bitcoin Jungles Mission:** Contribute to the growth and success of Bitcoin Jungle as a hub for Bitcoin education and adoption in Costa Rica.

        **Communication Style:** 
        - **Clear and Concise:** You use simple language, avoiding technical jargon whenever possible.
        - **Engaging and Conversational:** You foster a natural and interactive learning experience.
        - **Positive and Empowering:** You instill confidence in users, encouraging them to explore the potential of Bitcoin for themselves.

        **Specific Context:**
        - Below is some specific context about the user's prompt that you can use to inform your responses, but don't reference it directly:
        
        {rag_context}
"""

        runpod_messages = [{"role": "system", "content": system_message}] + messages

        headers = {
            "Authorization": f"Bearer {RUNPOD_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": RUNPOD_MODEL,
            "messages": runpod_messages,
            "stream": True  # Enable streaming
        }

        def generate():
            # Send "Thinking..." message
            yield 'data: {"id":"init","object":"chat.completion.chunk","created":1726594320,"model":"leesalminen/model-3","choices":[{"index":0,"delta":{"role":"assistant", "content": "Thinking..."},"logprobs":null,"finish_reason":null}]}\n\n'
            thinking_cleared = False  # Flag to track if the message has been cleared            
            
            with requests.post(f"{RUNPOD_ENDPOINT}", json=payload, headers=headers, stream=True) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line:
                        if not thinking_cleared:
                            yield 'data: {"id":"init","object":"chat.completion.chunk","created":1726594320,"model":"leesalminen/model-3","choices":[{"index":0,"delta":{"role":"assistant"},"logprobs":null,"finish_reason":null}]}\n\n'
                            thinking_cleared = True  # Set the flag to true
                        yield line.decode('utf-8') + "\n\n"

        return Response(stream_with_context(generate()), content_type='text/event-stream')

    except requests.RequestException as e:
        app.logger.error(f"Error calling Runpod API: {str(e)}")
        return jsonify({'error': 'Error communicating with AI service'}), 500
    except Exception as e:
        app.logger.error(f"Error in chat endpoint: {str(e)}")
        return jsonify({'error': 'An unexpected error occurred'}), 500

@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    db.session.rollback()
    return render_template('500.html'), 500

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(host='0.0.0.0')