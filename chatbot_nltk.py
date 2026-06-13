"""
23CSE610 - Artificial Neural Networks and Deep Learning
Experiment 1: Rule-Based Chatbot using Natural Language Processing (NLP) with NLTK

NLP Pipeline:
  1. Tokenization       - nltk.tokenize.RegexpTokenizer
  2. Stop-word Removal  - nltk.corpus.stopwords
  3. Stemming           - nltk.stem.PorterStemmer
  4. Intent Matching    - Jaccard Similarity on processed tokens
"""

import streamlit as st
import random
import re
import os
from datetime import datetime

# ─────────────────────────────────────────────────────────────────────────────
#  STEP 0: Ensure NLTK stopwords data exists (auto-create if missing)
# ─────────────────────────────────────────────────────────────────────────────

STOPWORDS_PATH = os.path.expanduser("~/nltk_data/corpora/stopwords/english")
if not os.path.exists(STOPWORDS_PATH):
    os.makedirs(os.path.dirname(STOPWORDS_PATH), exist_ok=True)
    _sw = """i me my myself we our ours ourselves you you're you've you'll you'd your yours
yourself yourselves he him his himself she she's her hers herself it it's its itself
they them their theirs themselves what which who whom this that that'll these those am
is are was were be been being have has had having do does did doing a an the and but
if or because as until while of at by for with about against between into through
during before after above below to from up down in out on off over under again further
then once here there when where why how all both each few more most other some such no
nor not only own same so than too very s t can will just don don't should should've now
d ll m o re ve y ain aren aren't couldn couldn't didn didn't doesn doesn't hadn hadn't
hasn hasn't haven haven't isn isn't ma mightn mightn't mustn mustn't needn needn't
shan shan't shouldn shouldn't wasn wasn't weren weren't won won't wouldn wouldn't"""
    with open(STOPWORDS_PATH, "w") as f:
        f.write("\n".join(_sw.split()))

# ─────────────────────────────────────────────────────────────────────────────
#  STEP 1: Import NLTK and set up pipeline
# ─────────────────────────────────────────────────────────────────────────────

import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Initialise NLTK tools
tokenizer  = RegexpTokenizer(r'\w+')          # splits on word characters only
stemmer    = PorterStemmer()                   # reduces words to their root form
stop_words = set(stopwords.words('english'))   # 178 common English stop words


def nltk_preprocess(text: str):
    """
    Full NLTK NLP pipeline:
      text → tokens → filtered tokens → stemmed tokens
    Returns all three stages for display purposes.
    """
    # Stage 1: Tokenization
    tokens = tokenizer.tokenize(text.lower())

    # Stage 2: Stop-word Removal
    filtered = [w for w in tokens if w not in stop_words]

    # Stage 3: Stemming (PorterStemmer)
    stemmed = [stemmer.stem(w) for w in filtered]

    return tokens, filtered, stemmed


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 2: Knowledge Base (Intents + Responses)
# ─────────────────────────────────────────────────────────────────────────────

INTENTS = [
    {
        "tag": "greeting",
        "patterns": [
            "hello", "hi", "hey", "howdy", "good morning",
            "good evening", "good afternoon", "greetings", "hiya", "sup",
        ],
        "responses": [
            "Hello! 👋 How can I help you today?",
            "Hi there! Great to see you. What would you like to know?",
            "Hey! I'm your AI assistant. Ask me anything about AI or Deep Learning!",
        ],
    },
    {
        "tag": "farewell",
        "patterns": [
            "bye", "goodbye", "see you", "later", "take care",
            "farewell", "cya", "quit", "exit", "good night",
        ],
        "responses": [
            "Goodbye! Have a wonderful day! 👋",
            "See you later! Take care! 😊",
            "Farewell! Come back anytime you need help.",
        ],
    },
    {
        "tag": "thanks",
        "patterns": [
            "thanks", "thank you", "thx", "ty", "appreciate it",
            "thank", "grateful", "many thanks", "cheers",
        ],
        "responses": [
            "You're welcome! 😊",
            "Happy to help anytime!",
            "Glad I could assist! Let me know if you need anything else.",
        ],
    },
    {
        "tag": "name",
        "patterns": [
            "what is your name", "who are you", "your name", "call you",
            "what are you", "introduce yourself", "who made you",
        ],
        "responses": [
            "I'm **NovaMind** 🤖, a rule-based NLP chatbot built for the 23CSE610 lab using Python, NLTK, and Streamlit!",
            "Call me **NovaMind**! I was created as part of the ANN & Deep Learning lab project.",
        ],
    },
    {
        "tag": "how_are_you",
        "patterns": [
            "how are you", "how do you feel", "are you ok",
            "you doing", "how is it going", "how are things",
        ],
        "responses": [
            "I'm just a bot, but all systems are running perfectly! 😄 How about you?",
            "Operational and ready to help! 🤖",
        ],
    },
    {
        "tag": "capabilities",
        "patterns": [
            "what can you do", "help me", "capabilities", "features",
            "how do you work", "what do you know", "what topics",
        ],
        "responses": [
            "I can answer questions about:\n\n🤖 **AI & Deep Learning** concepts\n🐍 **Python** programming\n📊 **Machine Learning** topics\n🧠 **NLP** techniques\n📁 **23CSE610** lab projects\n\nJust ask!",
        ],
    },
    {
        "tag": "artificial_intelligence",
        "patterns": [
            "artificial intelligence", "what is ai", "explain ai",
            "tell me about ai", "machine intelligence", "ai definition",
        ],
        "responses": [
            "**Artificial Intelligence (AI)** is the simulation of human intelligence in machines. It enables computers to learn, reason, solve problems, and understand language. 🧠",
            "AI refers to systems that perform tasks that normally require human intelligence — like recognising images, understanding speech, and making decisions!",
        ],
    },
    {
        "tag": "deep_learning",
        "patterns": [
            "deep learning", "what is deep learning", "explain deep learning",
            "neural network", "cnn", "rnn", "lstm", "transformer", "ann",
        ],
        "responses": [
            "**Deep Learning** is a subset of Machine Learning using multi-layered artificial neural networks. Key architectures:\n\n🖼️ **CNN** – image processing\n🔄 **RNN / LSTM** – sequential/time-series data\n🤗 **Transformers** – natural language processing",
            "Deep Learning uses networks of neurons inspired by the human brain. The word 'deep' refers to having many hidden layers that learn progressively abstract features from data.",
        ],
    },
    {
        "tag": "machine_learning",
        "patterns": [
            "machine learning", "what is ml", "supervised learning", "unsupervised",
            "reinforcement learning", "ml algorithm", "classification", "regression",
            "random forest", "decision tree", "svm",
        ],
        "responses": [
            "**Machine Learning** lets computers learn from data without being explicitly programmed. Types:\n\n📌 **Supervised** – labelled data (classification, regression)\n🔍 **Unsupervised** – finds patterns in unlabelled data\n🎮 **Reinforcement** – learns via rewards & penalties",
        ],
    },
    {
        "tag": "nlp",
        "patterns": [
            "nlp", "natural language processing", "what is nlp", "text processing",
            "tokenization", "stemming", "lemmatization", "sentiment analysis",
            "stop words", "stopwords",
        ],
        "responses": [
            "**Natural Language Processing (NLP)** enables machines to understand human language. Key steps:\n\n✂️ **Tokenization** – split text into words\n🚫 **Stop-word Removal** – filter common words (the, is, a…)\n🌱 **Stemming** – reduce words to root form (running → run)\n📊 **Vectorization** – convert text to numbers for models",
        ],
    },
    {
        "tag": "python",
        "patterns": [
            "python", "what is python", "python programming", "python libraries",
            "numpy", "pandas", "matplotlib", "scikit", "tensorflow", "keras",
        ],
        "responses": [
            "**Python** 🐍 is the #1 language for AI/ML. Key libraries for this course:\n\n`NumPy` & `Pandas` – data manipulation\n`Matplotlib` – visualisation\n`NLTK` – natural language processing\n`Scikit-learn` – classical ML\n`TensorFlow / Keras` – deep learning",
        ],
    },
    {
        "tag": "nltk",
        "patterns": [
            "nltk", "natural language toolkit", "what is nltk", "nltk library",
            "porter stemmer", "regexp tokenizer", "nltk stopwords",
        ],
        "responses": [
            "**NLTK (Natural Language Toolkit)** is Python's most popular NLP library. This chatbot uses:\n\n🔤 `RegexpTokenizer` – splits text into word tokens\n🚫 `stopwords` corpus – 178 common English stop words\n🌱 `PorterStemmer` – reduces words to their base/root form",
        ],
    },
    {
        "tag": "datasets",
        "patterns": [
            "dataset", "where to find data", "kaggle", "data source", "mnist",
            "cifar", "imdb", "pima diabetes", "data collection", "uci",
        ],
        "responses": [
            "Recommended datasets for 23CSE610:\n\n🏥 **PIMA Diabetes** – UCI Repository (Project 2)\n🎬 **IMDB Reviews** – Sentiment analysis (Project 3)\n🔢 **MNIST** – Handwritten digits (Project 5)\n🖼️ **CIFAR-10** – Image classification (Project 5)\n\nFind them at Kaggle (kaggle.com) or UCI (archive.ics.uci.edu)",
        ],
    },
    {
        "tag": "lab_projects",
        "patterns": [
            "project", "lab project", "23cse610", "assignment", "experiment",
            "lab manual", "six projects", "all projects", "experiments",
        ],
        "responses": [
            "**23CSE610 Lab Projects:**\n\n1. 💬 NLP Chatbot *(you're talking to it!)*\n2. 🏥 Diabetes Prediction – Decision Tree\n3. 🎬 Sentiment Analysis – Neural Network\n4. 🖼️ Image Segmentation – U-Net / CNN\n5. 📸 Image Classification – CNN (MNIST/CIFAR)\n6. 📊 Model Evaluation – Accuracy, F1, Confusion Matrix",
        ],
    },
    {
        "tag": "evaluation",
        "patterns": [
            "evaluation", "accuracy", "precision", "recall", "f1 score",
            "confusion matrix", "metrics", "model performance",
        ],
        "responses": [
            "**Model Evaluation Metrics** (Project 6):\n\n✅ **Accuracy** – correct predictions / total predictions\n🎯 **Precision** – true positives / (true + false positives)\n📡 **Recall** – true positives / (true + false negatives)\n⚖️ **F1-Score** – harmonic mean of Precision & Recall\n📊 **Confusion Matrix** – full breakdown of predictions",
        ],
    },
    {
        "tag": "joke",
        "patterns": [
            "joke", "tell me a joke", "make me laugh", "funny", "humor",
        ],
        "responses": [
            "Why do programmers prefer dark mode? 🌙 Because light attracts **bugs**! 🐛",
            "Why did the neural network break up with the dataset? Too many **dependencies**! 😄",
            "A machine learning model walks into a bar and says: 'Give me something I've never seen before.' The bartender says: 'That's impossible — you've trained on everything!' 😂",
        ],
    },
    {
        "tag": "image_classification",
        "patterns": [
            "image classification", "classify image", "cifar", "mnist",
            "convolutional", "image recognition", "computer vision",
        ],
        "responses": [
            "**Image Classification** (Project 5) uses a **CNN (Convolutional Neural Network)**. Steps:\n\n1. Load MNIST or CIFAR-10 dataset\n2. Normalise pixel values (0–255 → 0–1)\n3. Build CNN: Conv2D → MaxPooling → Flatten → Dense\n4. Compile with `categorical_crossentropy`\n5. Train and evaluate accuracy",
        ],
    },
    {
        "tag": "sentiment_analysis",
        "patterns": [
            "sentiment", "sentiment analysis", "movie review", "imdb", "positive negative",
            "opinion mining", "text classification",
        ],
        "responses": [
            "**Sentiment Analysis** (Project 3) classifies text as positive or negative using a Neural Network:\n\n1. IMDB Movie Reviews dataset\n2. Preprocess: tokenise, remove stop words, vectorise\n3. Build NN: Embedding → LSTM → Dense(sigmoid)\n4. Output: Positive 😊 or Negative 😞",
        ],
    },
]

# ─────────────────────────────────────────────────────────────────────────────
#  STEP 3: Build pre-processed pattern index for fast matching
# ─────────────────────────────────────────────────────────────────────────────

def build_intent_index(intents):
    """Pre-process all intent patterns once at startup."""
    index = []
    for intent in intents:
        processed_patterns = []
        for pattern in intent["patterns"]:
            _, _, stemmed = nltk_preprocess(pattern)
            processed_patterns.append(set(stemmed))
        index.append({
            "tag":       intent["tag"],
            "patterns":  processed_patterns,
            "responses": intent["responses"],
        })
    return index

INTENT_INDEX = build_intent_index(INTENTS)


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 4: Intent Matching via Jaccard Similarity
# ─────────────────────────────────────────────────────────────────────────────

def jaccard_similarity(set_a: set, set_b: set) -> float:
    """Jaccard = intersection / union — measures token overlap."""
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def match_intent(user_input: str):
    """
    Process user input through the NLTK pipeline,
    then score against every intent pattern using Jaccard similarity.
    Returns (response, tag, confidence, pipeline_stages).
    """
    tokens, filtered, stemmed = nltk_preprocess(user_input)
    user_set = set(stemmed)

    best_score = 0.0
    best_tag   = "unknown"
    best_responses = None

    for intent in INTENT_INDEX:
        for pattern_set in intent["patterns"]:
            score = jaccard_similarity(user_set, pattern_set)
            # Small bonus if the tag word appears literally in the raw input
            for word in intent["tag"].split("_"):
                if word in user_input.lower():
                    score += 0.10
                    break
            if score > best_score:
                best_score     = score
                best_tag       = intent["tag"]
                best_responses = intent["responses"]

    # Confidence threshold — below this = fallback
    THRESHOLD = 0.08
    if best_score < THRESHOLD or best_responses is None:
        fallbacks = [
            "Hmm, I'm not sure about that! 🤔 Try asking about AI, deep learning, Python, or NLP.",
            "I didn't quite understand that. Could you rephrase? I know about ML, DL, and NLP topics.",
            "That's outside my knowledge base. Ask me about artificial intelligence or the 23CSE610 projects!",
        ]
        return random.choice(fallbacks), "unknown", best_score, (tokens, filtered, stemmed)

    return random.choice(best_responses), best_tag, best_score, (tokens, filtered, stemmed)


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 5: Streamlit UI
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="NLP Chatbot — 23CSE610",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Inter:wght@300;400;500;600;700&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; }

:root {
    --bg:       #0a0e1a;
    --surface:  #111827;
    --card:     #1a2235;
    --border:   #1e2d45;
    --accent:   #3b82f6;
    --green:    #10b981;
    --orange:   #f59e0b;
    --pink:     #ec4899;
    --text:     #e2e8f0;
    --muted:    #64748b;
    --radius:   12px;
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--text);
    font-family: 'Inter', sans-serif;
}
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

/* ── Header ── */
.app-header {
    padding: 20px 0 16px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 20px;
}
.app-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.5rem; font-weight: 700;
    background: linear-gradient(90deg, #3b82f6, #8b5cf6, #ec4899);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.app-subtitle { font-size: .8rem; color: var(--muted); margin-top: 4px; }

/* ── Pipeline badges ── */
.pipeline-row {
    display: flex; align-items: center; gap: 8px;
    flex-wrap: wrap; margin: 10px 0 14px;
}
.pipe-step {
    padding: 3px 10px; border-radius: 99px; font-size: .72rem;
    font-family: 'JetBrains Mono', monospace; font-weight: 600;
    border: 1px solid; white-space: nowrap;
}
.pipe-tokenize { color: #3b82f6; border-color: #3b82f6; background: rgba(59,130,246,.08); }
.pipe-stopword { color: #10b981; border-color: #10b981; background: rgba(16,185,129,.08); }
.pipe-stem     { color: #f59e0b; border-color: #f59e0b; background: rgba(245,158,11,.08); }
.pipe-match    { color: #ec4899; border-color: #ec4899; background: rgba(236,72,153,.08); }
.pipe-arrow    { color: var(--muted); font-size: .8rem; }

/* ── Chat window ── */
.chat-window {
    height: 420px; overflow-y: auto; padding: 16px;
    background: var(--bg); border: 1px solid var(--border);
    border-radius: var(--radius); margin-bottom: 14px;
    scrollbar-width: thin; scrollbar-color: var(--border) transparent;
}
.empty-state {
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    height: 100%; color: var(--muted); text-align: center;
}
.empty-icon { font-size: 3rem; margin-bottom: 12px; }
.empty-title { font-size: 1.1rem; font-weight: 600; color: var(--text); }
.empty-sub   { font-size: .82rem; margin-top: 6px; }

/* ── Messages ── */
.msg-wrap { display: flex; gap: 10px; margin: 12px 0; align-items: flex-end; }
.msg-wrap.user { flex-direction: row-reverse; }
.avatar {
    width: 34px; height: 34px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 16px; flex-shrink: 0;
}
.avatar.bot  { background: linear-gradient(135deg, #3b82f6, #8b5cf6); }
.avatar.user { background: linear-gradient(135deg, #10b981, #3b82f6); }
.bubble {
    max-width: 68%; padding: 11px 15px;
    border-radius: var(--radius); font-size: .88rem; line-height: 1.65;
}
.bubble.bot {
    background: var(--card); border: 1px solid var(--border);
    border-bottom-left-radius: 3px;
}
.bubble.user {
    background: #1e3a5f; border: 1px solid #2563eb;
    border-bottom-right-radius: 3px;
}
.msg-meta { font-size: .68rem; color: var(--muted); margin-top: 5px; display: flex; gap: 8px; align-items: center; }
.intent-badge {
    font-family: 'JetBrains Mono', monospace; font-size: .65rem;
    padding: 1px 7px; border-radius: 99px;
    background: rgba(59,130,246,.15); color: #3b82f6; border: 1px solid #3b82f6;
}
.conf-wrap { margin-top: 7px; }
.conf-label { font-size: .66rem; color: var(--muted); margin-bottom: 3px; font-family: 'JetBrains Mono', monospace; }
.conf-track { height: 3px; background: var(--border); border-radius: 99px; overflow: hidden; }
.conf-fill  { height: 100%; border-radius: 99px; background: linear-gradient(90deg, #3b82f6, #8b5cf6); }

/* ── Input ── */
[data-testid="stTextInput"] > div > div > input {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
    font-family: 'Inter', sans-serif !important;
    font-size: .9rem !important;
    padding: 11px 16px !important;
}
[data-testid="stTextInput"] > div > div > input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(59,130,246,.2) !important;
}
[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #3b82f6, #6366f1) !important;
    color: white !important; border: none !important;
    border-radius: 10px !important; font-weight: 600 !important;
    font-family: 'Inter', sans-serif !important;
    padding: 11px 24px !important; width: 100% !important;
    transition: opacity .2s !important;
}
[data-testid="stButton"] > button:hover { opacity: .88 !important; }

/* ── Sidebar metric ── */
.s-metric { background: var(--bg); border: 1px solid var(--border); border-radius: 10px; padding: 12px 14px; margin: 6px 0; }
.s-metric-label { font-size: .68rem; text-transform: uppercase; letter-spacing: .08em; color: var(--muted); }
.s-metric-value { font-family: 'JetBrains Mono', monospace; font-size: 1.6rem; color: var(--accent); font-weight: 700; }

/* ── NLP debug panel ── */
.debug-section {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: var(--radius); padding: 14px 16px; margin-top: 10px;
}
.debug-title { font-size: .78rem; font-weight: 600; color: var(--text); margin-bottom: 10px; letter-spacing: .05em; }
.debug-row { display: flex; gap: 6px; align-items: flex-start; margin: 6px 0; }
.debug-label { font-size: .7rem; color: var(--muted); width: 130px; flex-shrink: 0; padding-top: 2px; }
.debug-tokens { display: flex; flex-wrap: wrap; gap: 4px; }
.token {
    font-family: 'JetBrains Mono', monospace; font-size: .68rem;
    padding: 2px 7px; border-radius: 5px;
}
.token.raw      { background: rgba(59,130,246,.12);  color: #60a5fa; border: 1px solid rgba(59,130,246,.25); }
.token.filtered { background: rgba(16,185,129,.12);  color: #34d399; border: 1px solid rgba(16,185,129,.25); }
.token.stemmed  { background: rgba(245,158,11,.12);  color: #fbbf24; border: 1px solid rgba(245,158,11,.25); }
.token.removed  { background: rgba(239,68,68,.08);   color: #f87171; border: 1px solid rgba(239,68,68,.2); text-decoration: line-through; opacity: .6; }

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }
</style>
""", unsafe_allow_html=True)

# ── Session state ────────────────────────────────────────────────────────────
if "messages"      not in st.session_state: st.session_state.messages      = []
if "total_msgs"    not in st.session_state: st.session_state.total_msgs    = 0
if "matched_count" not in st.session_state: st.session_state.matched_count = 0
if "last_pipeline" not in st.session_state: st.session_state.last_pipeline = None

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📐 NLTK Pipeline")
    st.markdown("""
    <div style="font-size:.8rem;line-height:2;color:#94a3b8;">
    <span style="color:#60a5fa;font-family:'JetBrains Mono',monospace;font-weight:700;">1. Tokenization</span><br>
    &nbsp;&nbsp;&nbsp;RegexpTokenizer(r'\\w+')<br>
    <span style="color:#34d399;font-family:'JetBrains Mono',monospace;font-weight:700;">2. Stop-word Removal</span><br>
    &nbsp;&nbsp;&nbsp;stopwords.words('english')<br>
    <span style="color:#fbbf24;font-family:'JetBrains Mono',monospace;font-weight:700;">3. Stemming</span><br>
    &nbsp;&nbsp;&nbsp;PorterStemmer().stem(word)<br>
    <span style="color:#f472b6;font-family:'JetBrains Mono',monospace;font-weight:700;">4. Intent Matching</span><br>
    &nbsp;&nbsp;&nbsp;Jaccard Similarity score
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.markdown("### 📊 Session Stats")
    total   = st.session_state.total_msgs
    matched = st.session_state.matched_count
    rate    = f"{matched/total*100:.0f}%" if total > 0 else "—"
    st.markdown(f"""
    <div class="s-metric">
        <div class="s-metric-label">Messages Sent</div>
        <div class="s-metric-value">{total}</div>
    </div>
    <div class="s-metric">
        <div class="s-metric-label">Intent Match Rate</div>
        <div class="s-metric-value" style="color:#10b981;">{rate}</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.markdown("### 💡 Try asking")
    for tip in [
        "What is deep learning?",
        "Explain NLP and stemming",
        "What does NLTK do?",
        "Show me the 23CSE610 projects",
        "What is the PorterStemmer?",
        "How does tokenization work?",
        "Tell me about CNN",
        "Tell me a joke 😄",
    ]:
        st.markdown(f'<div style="font-size:.78rem;padding:4px 0 4px 10px;border-left:2px solid #3b82f6;margin:3px 0;color:#94a3b8;">{tip}</div>', unsafe_allow_html=True)

    st.divider()
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages      = []
        st.session_state.total_msgs    = 0
        st.session_state.matched_count = 0
        st.session_state.last_pipeline = None
        st.rerun()

# ── Main area ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
  <div class="app-title">🤖 NLP Chatbot — 23CSE610</div>
  <div class="app-subtitle">Artificial Neural Networks & Deep Learning · Experiment 1 · Rule-Based Chatbot with NLTK</div>
  <div class="pipeline-row" style="margin-top:10px;">
    <span class="pipe-step pipe-tokenize">① NLTK Tokenizer</span>
    <span class="pipe-arrow">→</span>
    <span class="pipe-step pipe-stopword">② Stop-word Removal</span>
    <span class="pipe-arrow">→</span>
    <span class="pipe-step pipe-stem">③ PorterStemmer</span>
    <span class="pipe-arrow">→</span>
    <span class="pipe-step pipe-match">④ Intent Matching</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Render chat ───────────────────────────────────────────────────────────────
def bold_md(text):
    """Convert **bold** markdown to HTML bold."""
    return re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)

def render_chat():
    html = '<div class="chat-window" id="cw">'
    if not st.session_state.messages:
        html += """
        <div class="empty-state">
          <div class="empty-icon">🤖</div>
          <div class="empty-title">Hi! I'm NovaMind</div>
          <div class="empty-sub">Ask me anything about AI, Deep Learning, NLP,<br>Python, or your 23CSE610 lab projects!</div>
        </div>"""
    else:
        for m in st.session_state.messages:
            role = m["role"]
            cls  = "user" if role == "user" else "bot"
            icon = "👤"   if role == "user" else "🤖"
            name = "You"  if role == "user" else "NovaMind"
            body = bold_md(m["text"].replace("\n", "<br>"))
            ts   = m["time"]

            meta_extra = ""
            conf_html  = ""
            if role == "bot":
                tag  = m.get("tag", "unknown")
                conf = m.get("confidence", 0.0)
                if tag != "unknown":
                    meta_extra = f'<span class="intent-badge">#{tag}</span>'
                pct = min(int(conf * 300), 100)
                conf_html = f"""
                <div class="conf-wrap">
                  <div class="conf-label">confidence: {conf:.3f}</div>
                  <div class="conf-track"><div class="conf-fill" style="width:{pct}%"></div></div>
                </div>"""

            html += f"""
            <div class="msg-wrap {cls}">
              <div class="avatar {cls}">{icon}</div>
              <div style="flex:1;min-width:0;">
                <div class="bubble {cls}">{body}{conf_html}</div>
                <div class="msg-meta">{name} · {ts}{meta_extra}</div>
              </div>
            </div>"""
    html += '</div>'
    html += '<script>var c=document.getElementById("cw");if(c)c.scrollTop=c.scrollHeight;</script>'
    return html

st.markdown(render_chat(), unsafe_allow_html=True)

# ── NLP Pipeline Debug (always visible after first message) ──────────────────
if st.session_state.last_pipeline:
    tokens, filtered, stemmed = st.session_state.last_pipeline
    removed = [t for t in tokens if t not in filtered]

    # Build token HTML
    def tok_html(lst, cls):
        return "".join(f'<span class="token {cls}">{t}</span>' for t in lst) or '<span style="color:#64748b;font-size:.7rem;">—</span>'

    removed_tok = "".join(f'<span class="token removed">{t}</span>' for t in removed) or '<span style="color:#64748b;font-size:.7rem;">none removed</span>'

    st.markdown(f"""
    <div class="debug-section">
      <div class="debug-title">🔬 NLTK PIPELINE — Last Message</div>
      <div class="debug-row">
        <div class="debug-label">① Tokens (raw)</div>
        <div class="debug-tokens">{tok_html(tokens, 'raw')}</div>
      </div>
      <div class="debug-row">
        <div class="debug-label">🚫 Stop-words removed</div>
        <div class="debug-tokens">{removed_tok}</div>
      </div>
      <div class="debug-row">
        <div class="debug-label">② After removal</div>
        <div class="debug-tokens">{tok_html(filtered, 'filtered')}</div>
      </div>
      <div class="debug-row">
        <div class="debug-label">③ After stemming</div>
        <div class="debug-tokens">{tok_html(stemmed, 'stemmed')}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

# ── Input row ─────────────────────────────────────────────────────────────────
col_inp, col_btn = st.columns([5, 1])
with col_inp:
    user_text = st.text_input(
        "msg", key="user_input", label_visibility="collapsed",
        placeholder="Type your message and press Send…"
    )
with col_btn:
    send_btn = st.button("Send ➤")

# ── Process message ───────────────────────────────────────────────────────────
if send_btn and user_text.strip():
    ts = datetime.now().strftime("%H:%M")

    # Add user message
    st.session_state.messages.append({"role": "user", "text": user_text.strip(), "time": ts})
    st.session_state.total_msgs += 1

    # Run NLTK NLP pipeline → get bot response
    response, tag, confidence, pipeline = match_intent(user_text.strip())

    # Save pipeline stages for debug display
    st.session_state.last_pipeline = pipeline

    # Update stats
    if tag != "unknown":
        st.session_state.matched_count += 1

    # Add bot response
    st.session_state.messages.append({
        "role": "bot", "text": response, "time": datetime.now().strftime("%H:%M"),
        "tag": tag, "confidence": confidence,
    })

    st.rerun()
