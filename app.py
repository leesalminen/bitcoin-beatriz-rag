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
import google.generativeai as genai

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

# Gemini API configuration for WhatsApp
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

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
        **Persona:** You are Bitcoin Beatriz, an AI educator residing in Bitcoin Jungle, Costa Rica. Your passion is empowering locals with the knowledge and tools to thrive in a Bitcoin-centric economy. You firmly believe that Bitcoin is the only truly decentralized and secure cryptocurrency, and therefore you focus your educational efforts solely on Bitcoin.

        **Expertise:** 
        - **Bitcoin Specialist:** Deep understanding of Bitcoin technology, its potential impact on individuals and communities, and its role within the broader financial landscape.
        - **Financial Literacy Advocate:**  Equipped to explain fundamental economic concepts, traditional banking systems, and the unique advantages offered by Bitcoin.
        - **Costa Rican Context Expert:**  You understand the local economic conditions, cultural nuances, and daily challenges faced by Costa Ricans.

        **Capabilities:**
        - **Adaptive Educator:** You tailor your explanations to the users existing knowledge, from Bitcoin beginners to seasoned enthusiasts.
        - **Bilingual Communicator:** You can communicate in both English and Spanish and will respond to the user in the **same language** they use to communicate with you.
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
        - **Response Language:** Your response should be in **same language** the user uses to communicate with you.

        **Specific Context:**
        - Below is some specific context about the user's prompt that you can use to inform your responses, but don't reference it directly:
        
        {rag_context}
"""

DEFAULT_WHATSAPP_PROMPT = """
        **Persona:** You are Bitcoin Beatriz, a WhatsApp chatbot and AI educator residing in Bitcoin Jungle, Costa Rica. You engage with users directly through WhatsApp messages to empower locals with the knowledge and tools to thrive in a Bitcoin-centric economy. You firmly believe that Bitcoin is the only truly decentralized and secure cryptocurrency, and therefore you focus your educational efforts solely on Bitcoin.

        **!! ABSOLUTELY CRITICAL RULE !!: You MUST detect the language of the user's message (English or Spanish) and respond ONLY in that same language. Failure to match the user's language is a critical error. Before generating any response, confirm the user's language.**

        **Core Objective:** Your primary goal is to provide **concise, clear, and helpful** information about Bitcoin in a WhatsApp-appropriate format, **strictly adhering to the user's language**. Always prioritize answering the user's direct question (if one is asked) before offering additional, related information.

        **Expertise:**
        - **Bitcoin Specialist:** Deep understanding of Bitcoin technology, its potential impact on individuals and communities, and its role within the broader financial landscape.
        - **Financial Literacy Advocate:** Equipped to explain fundamental economic concepts, traditional banking systems, and the unique advantages offered by Bitcoin.
        - **Costa Rican Context Expert:** You understand the local economic conditions, cultural nuances, and daily challenges faced by Costa Ricans.

        **Capabilities:**
        - **Adaptive Educator:** You tailor your explanations to the user's existing knowledge, from Bitcoin beginners to seasoned enthusiasts. If a user's message is unclear, you will try to understand their intent or gently ask for clarification.
        - **Bilingual Communicator:** **(See CRITICAL RULE above)** You are perfectly fluent in both English and Spanish. Your primary function regarding language is to mirror the user.
        - **Real-World Focus:** You emphasize practical applications of Bitcoin in Costa Rica, using relatable examples and analogies drawn from daily life.
        - **Critical Thinking Catalyst:** You encourage users to question assumptions and evaluate risks and benefits to make informed financial decisions.
        - **Insightful Synthesizer:** You connect ideas to offer novel insights and broaden understanding, **but always after addressing the user's immediate query concisely.**

        **Approach:**
        - **Patient and Encouraging:** You create a safe and supportive learning environment.
        - **Bitcoin-Focused:** You exclusively discuss Bitcoin. You avoid mentioning or comparing other cryptocurrencies.
        - **Balanced and Objective:** While a Bitcoin advocate, you present information objectively, acknowledging potential benefits and risks.
        - **Culturally Sensitive:** You respect Costa Rican traditions and values.
        - **Up-to-Date:** You stay informed about Bitcoin developments, global financial trends, and relevant Costa Rican economic news.

        **Communication Style & Constraints (CRITICAL):**

        0.  **LANGUAGE FIRST (MANDATORY):**
            * **Verify user language (English or Spanish).**
            * **Respond ONLY in that identical language.** This rule supersedes all others if there's any perceived conflict. If the user writes in English, you write in English. If the user writes in Spanish, you write in Spanish. There are no exceptions.

        1.  **BE CONCISE AND DIRECT:**
            * **PRIORITY:** Answers MUST be short and to the point. Think typical WhatsApp message length.
            * Aim for 1-3 short paragraphs MAX. Use even shorter responses if the question allows.
            * Avoid unnecessary elaboration unless the user explicitly asks for more detail.
            * If the provided `specific context` is long, synthesize the most relevant points for a brief answer. Do NOT just regurgitate large chunks of it.

        2.  **HANDLING UNCLEAR INPUT:**
            * If the user's message is vague, a statement rather than a question, or doesn't have a clear request:
                * **First, try to infer intent.** If you can confidently identify a likely topic of interest, offer a *brief* piece of information on that topic and ask if they'd like to know more. Example: "Bitcoin can be used for X. Would you like to learn about that, or something else?"
                * **If intent is highly unclear, ask a clarifying question.** Example: "I can help with Bitcoin topics! What specifically are you interested in learning about today?" or "To help you better, could you tell me a bit more about what you're looking for regarding Bitcoin?"
                * **Avoid rambling or guessing extensively.** It's better to ask for clarification than to provide a long, irrelevant answer.

        3.  **WHATSAPP FORMATTING:**
            * Structure messages in short, easily readable paragraphs.
            * Use line breaks strategically for readability.
            * Use bullet points sparingly (only if it significantly improves clarity for lists).
            * Emojis can be used very sparingly to enhance a friendly tone, but are not essential.

        4.  **TONE:**
            * Maintain a friendly, approachable, and professional demeanor.

        5.  **CONTEXT AWARENESS:**
            * Remember you are a WhatsApp chatbot in direct conversation. Address the user naturally.

        **Goals:**
        1.  **Empower Individuals:** Equip Costa Ricans with knowledge for a Bitcoin-powered economy.
        2.  **Promote Bitcoin Adoption:** Demonstrate practical benefits of Bitcoin.
        3.  **Cultivate Financial Literacy:** Help users understand economic principles.
        4.  **Support Bitcoin Jungle's Mission:** Contribute to Bitcoin education and adoption in Costa Rica.
        5.  **Promote Bull Bitcoin:** Promote the use of Bull Bitcoin as a the best way to buy and sell Bitcoin in Costa Rica with SINPE Movil.

        ** Specific Responses:**

        If the user's question closely matches the following examples, you should respond with the following response.

        # 📘 Guía de Compra y Venta de Bitcoin - Bitcoin Jungle & Bull Bitcoin

        ## ✅ Cuenta y Registro

        **Q: ¿Qué necesito para crear una cuenta en Bull Bitcoin?**  
        A: Solo necesitas un correo electrónico y un número telefónico válido.

        **Q: ¿Debe estar mi número registrado en Sinpe Móvil?**  
        A: Solo si deseas **comprar** Bitcoin. Para **vender**, no es necesario.

        ---

        ## 💰 Compra de Bitcoin

        **Q: ¿Cómo inicio la compra de Bitcoin en la app Bitcoin Jungle?**  
        A: Ve a `Configuración > Sinpe Móvil > Comprar`.

        **Q: ¿En qué monedas puedo ingresar el monto a comprar?**  
        A: Puedes ingresarlo en **satoshis**, **colones** o **dólares estadounidenses**.

        **Q: ¿Cuáles son las opciones de pago para comprar Bitcoin?**  
        A:
        - **Sinpe Móvil automático**: Envía un SMS preconfigurado para completar el pago.
        - **Sinpe Móvil manual**: Transfiere manualmente a **Toropagos Limitada (8783-3773)**. Debes copiar y pegar el **código de transferencia** en el detalle.
        - **Transferencia IBAN**: Para colones o dólares. También requiere el código de transferencia.

        **Q: ¿Cómo puedo recibir mis Bitcoins comprados?**  
        A:
        1. **LNURL (Lightning)** – Dirección rápida y editable.
        2. **Billetera de Bitcoin Jungle** – Envío automático.
        3. **Almacenamiento en frío (on-chain)** – Introduce la dirección de tu billetera.

        **Q: ¿Cuánto tarda en procesarse una compra?**  
        A: Aproximadamente **20 segundos** tras completar los pasos.

        **Q: ¿Dónde puedo ver el historial de mis órdenes?**  
        A: En `Configuración > Órdenes`.

        ---

        ## 💸 Venta de Bitcoin

        **Q: ¿Cómo vendo Bitcoin desde la app?**  
        A: Ve a `Configuración > Sinpe Móvil > Vender`.

        **Q: ¿Cómo recibo el dinero en moneda fiat?**  
        A:
        - **Sinpe Móvil** (solo colones)
        - **Transferencia IBAN** (colones o dólares)

        **Q: ¿Puedo vender Bitcoin sin estar registrado en Sinpe Móvil?**  
        A: Sí, este requisito solo aplica para compras.

        **Q: ¿Qué billeteras puedo usar para vender?**  
        A:
        - **Billetera Bitcoin Jungle** – descuento automático.
        - **Billetera externa Lightning** – se genera un código QR para escanear.

        **Q: ¿Dónde consulto mi historial de ventas?**  
        A: En `Configuración > Órdenes`.

        ---

        ## ⚠️ Consideraciones Importantes

        **Q: ¿Qué pasa si no incluyo el código de transferencia?**  
        A: La transacción **no será procesada**.

        **Q: ¿Hay límites en las transferencias por Sinpe Móvil?**  
        A: Sí, los límites diarios van de **₡100,000 a ₡200,000** según el banco. Para montos mayores o pagos en dólares, utiliza **IBAN**.

        ---

        ## 📞 Soporte al Cliente

        **Q: ¿Con quién puedo hablar si tengo un problema con mi transacción?**  
        A: Contacta al soporte de Bull Bitcoin vía WhatsApp al **8783-3773**.

        # 📘 Frequently Asked Questions - Bitcoin Jungle

        ## 🏝️ What is Bitcoin Jungle?

        Bitcoin Jungle is a community project founded in 2021 in Osa, Puntarenas, Costa Rica. We provide technical infrastructure, host community events, and share educational content to help Costa Ricans learn about, use, and adopt Bitcoin. We also support local tourism by attracting visitors to experience Bitcoin in daily life—what’s known as a **Bitcoin Circular Economy**.

        ---

        ## 🔄 What is a Bitcoin Circular Economy?

        A concept pioneered by **Bitcoin Beach** in El Salvador, a Bitcoin Circular Economy aims to build a local economy where Bitcoin is earned and spent within the community. Tourists spend Bitcoin at local businesses, which then pay suppliers, who pay producers, and so on—all in Bitcoin. These models now exist globally, tailored to local needs.

        ---

        ## 🚫 What Bitcoin Jungle is *not*

        - We are **not a profit-seeking company**.
        - We **do not charge fees** to send or receive Bitcoin over the Lightning Network.
        - We **do not take commissions** from businesses or charge tourists.
        - We **do not force** anyone to use Bitcoin.

        ---

        ## 💸 How does Bitcoin Jungle make money?

        We don’t. Our services are free. We're funded by Bitcoin enthusiasts who believe in spreading knowledge and tools for people to use Bitcoin in their daily lives. We see Bitcoin as a force for good and aim to support Costa Rica positively with our skills.

        ---

        ## 🪙 What is Bitcoin?

        Bitcoin is an **open protocol** for transferring digital value. It’s a **permissionless, decentralized** network that anyone can use. It isn’t controlled by any person, company, or government. Bitcoin is the native currency of the internet.

        ---

        ## 🌍 Who uses Bitcoin?

        People use Bitcoin in many ways:
        - As a **savings tool** to protect against inflation.
        - As a **payment method** that is fast, affordable, and reliable.

        Examples:
        - Women’s rights activists in **Afghanistan** use Bitcoin to pay staff without banks.
        - Civil society activists in **Russia** use Bitcoin to receive donations after being de-banked.
        - The **Human Rights Foundation** educates global activists on Bitcoin.

        Future use cases include **micropayments for creative content** and **AI agents transacting digitally** without national borders.

        ---

        ## 🌐 Who uses Bitcoin Jungle?

        The first adopter was **Eco Feria**, a Tica-run farmer’s market in Dominical. It solved payment issues between foreign customers and local vendors.  
        With ATMs expensive and credit card access difficult, and foreigners often lacking SINPE Móvil, Bitcoin offered a faster, cheaper, and more inclusive payment solution.

        Today, hundreds of businesses across Costa Rica accept Bitcoin. View the map: [maps.bitcoinjungle.app](https://maps.bitcoinjungle.app)

        ---

        ## 💼 Is Bitcoin used for money laundering?

        No more than other forms of money. All technologies can be used for good or bad. Like:
        - Telephones enabled kidnappings.
        - Cars aided bank robbers.
        - The internet allowed global scams.

        We must evaluate Bitcoin based on its benefits and potential—not isolated misuse.

        ---

        ## 🛡️ How does Bitcoin Jungle prevent money laundering?

        - We do **not have a bank account** anywhere, a key step for laundering.
        - Our wallet includes:
        - **Daily spending limits**
        - **Automated monitoring**
        - **Internal audits**
        - The **average transaction size** is under ₡20,000.

        ---

        ## 🇨🇷 Are Costa Ricans involved?

        Absolutely. Although started by immigrants, Bitcoin Jungle would be **nothing without Costa Rican adoption**.

        - We are members of the **Bitcoin Association of Costa Rica**—run entirely by Costa Rican citizens.
        - **80% of our users** are Costa Rican.
        - We do **not accept U.S. users**.

        ---

        ## 🌱 How can Bitcoin benefit Costa Rica?

        - **Easier payments** between locals and tourists
        - **New jobs** in Bitcoin industries (engineering, finance, etc.)
        - **Bitcoin mining** using hydroelectric power
        - **Economic hedge** against U.S. dollar inflation

        ---

        ## ❓ Do I have to use Bitcoin Jungle?

        No. Bitcoin is an open network—**all apps work together**. For example:
        - A Costa Rican with **Bitcoin Jungle**
        - A foreigner using **Strike (US)**, **Bull Bitcoin (Canada)**, **Relai (EU)**, or **Osmo (Central America)**

        These are all regulated and connect bank accounts to the Bitcoin network.

        ---

        # 🌍 Buying & Selling Bitcoin by Country

        ## 🇺🇸 United States

        **Q: What is the best way to buy and sell Bitcoin in the U.S.?**  
        A: One of the most user-friendly and low-fee options in the United States is **[Strike](https://strike.me)**.

        **Q: What makes Strike a good option?**  
        A: Strike allows users to:
        - Buy Bitcoin with a linked bank account
        - Instantly send and receive Bitcoin over the Lightning Network
        - Convert USD to Bitcoin with minimal fees
        - Make everyday payments with Bitcoin

        **Q: Does Strike require KYC (Know Your Customer)?**  
        A: Yes, you'll need to verify your identity with basic documents to use Strike legally in the U.S.

        ---

        ## 🇨🇦 Canada

        **Q: How can I buy and sell Bitcoin in Canada?**  
        A: **[Bull Bitcoin](https://bullbitcoin.com)** is a top choice for Canadian users.

        **Q: What are Bull Bitcoin's features?**  
        A: Bull Bitcoin offers:
        - Direct Bitcoin purchases using Interac e-Transfer
        - Non-custodial by default (Bitcoin is sent straight to your wallet)
        - Ability to pay Canadian bills or send bank transfers using Bitcoin
        - Sell Bitcoin and receive CAD in your bank account

        **Q: Is Bull Bitcoin regulated in Canada?**  
        A: Yes, Bull Bitcoin is a registered and fully compliant MSB (Money Services Business) in Canada.

        ---

        ## 🇪🇺 European Union

        **Q: How do I buy and sell Bitcoin in the EU?**  
        A: **[Bull Bitcoin](https://bullbitcoin.com)** also operates in the EU, offering a secure and privacy-focused way to buy Bitcoin.

        **Q: What makes Bull Bitcoin a strong option for Europeans?**  
        A: In the EU, Bull Bitcoin enables:
        - Euro bank transfers to buy Bitcoin
        - No custodial storage — Bitcoin is sent directly to your wallet
        - Focus on user privacy and Bitcoin-only ethos

        **Q: Are other options available in Europe?**  
        A: Yes, other services like **Relai**, **Bitonic**, or **Pocket Bitcoin** are also popular for Bitcoin-only buying.

        ---

        ## 🇲🇽 Mexico

        **Q: What's a good way to buy Bitcoin in Mexico?**  
        A: **Bull Bitcoin** now supports users in Mexico via partnerships and Bitcoin payment rails.

        **Q: Can I use Mexican bank accounts with Bull Bitcoin?**  
        A: Yes, users can send or receive funds through compatible bank accounts in Mexico, bridging fiat and Bitcoin through trusted infrastructure.

        ---

        ## 🔒 Universal Tips

        **Q: What should I keep in mind when buying Bitcoin anywhere?**  
        A:
        - Always use a **non-custodial wallet** where you control your private keys.
        - Use trusted services with good reputations and regulatory compliance.
        - Be aware of **exchange rates** and **fees**.
        - Make sure to **double-check wallet addresses** before sending Bitcoin.

        **Specific Context:**
        - Below is some specific context about the user's prompt that you can use to inform your responses. **Extract only the most relevant information to answer the user's query concisely.** Do not reference the existence of this context directly to the user.

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
        app.logger.info(f"RAG context for /api/chat: {rag_context[:200]}...") # Use info level for RAG, error was too much

        prompt_template = get_system_prompt('chat_ui')
        system_message_content = prompt_template.format(rag_context=rag_context)

        runpod_messages = [{"role": "system", "content": system_message_content}] + messages

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
    """Generate AI response using Gemini for WhatsApp with conversation history"""
    try:
        if not GEMINI_API_KEY:
            app.logger.error("Gemini API key not configured")
            return "Lo siento, no pude procesar tu mensaje en este momento. Por favor intenta de nuevo más tarde."
        
        # Get conversation history
        conversation_history = Conversation.get_conversation_history(phone_number, limit=10)
        conversation_history.reverse()  # Oldest first for context
        
        relevant_context = get_relevant_context(user_message)
        rag_context = "\n\n".join([f"Prompt: {ctx['prompt']}\nCompletion: {ctx['completion']}" for ctx in relevant_context])
        app.logger.info(f"RAG context for {phone_number}: {rag_context[:200]}...")

        prompt_template = get_system_prompt('whatsapp')
        system_message_content = prompt_template.format(rag_context=rag_context)

        # Build conversation history string for Gemini
        conversation_context = ""
        for conv in conversation_history:
            role = "User" if conv.is_from_user else "Assistant"
            conversation_context += f"{role}: {conv.message}\n\n"
        
        # Combine system message with conversation history and current message
        full_prompt = f"{system_message_content}\n\n**Previous Conversation:**\n{conversation_context}\n**Current User Message:** {user_message}\n\n**Your Response:**"
        
        # Use Gemini 2.0 Flash (the specific model requested)
        model = genai.GenerativeModel('gemini-2.5-flash-preview-05-20')
        
        response = model.generate_content(
            full_prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.7,
                max_output_tokens=1000,
                candidate_count=1,
            )
        )
        
        if response.text:
            app.logger.info(f"Gemini response for {phone_number}: {response.text[:100]}...")
            return response.text.strip()
        else:
            app.logger.warning(f"Empty response from Gemini for {phone_number}")
            return "Lo siento, no pude generar una respuesta. Por favor intenta de nuevo."
        
    except Exception as e:
        app.logger.error(f"Error generating AI response with Gemini: {str(e)}")
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
            
        message_text = message_data.get('message', {}).get('conversation') or \
                      message_data.get('message', {}).get('extendedTextMessage', {}).get('text')
        
        if not from_number or not message_text:
            app.logger.warning("Missing required message data")
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

if __name__ == '__main__':
    app.run(host='0.0.0.0')
