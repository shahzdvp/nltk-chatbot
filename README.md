# 🤖 NLP Chatbot with Streamlit UI

A rule-based conversational chatbot built with Python, NLTK, and Streamlit. The bot understands user intent by preprocessing natural language input and matching it against a library of defined intents using **cosine similarity** on bag-of-words vectors.

---

## 📸 Demo

> Launch the app and chat in real time. The sidebar displays intent confidence scores after every message so you can see exactly how the bot made its decision.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- pip

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-username/chatbot.git
cd chatbot

# 2. (Optional) Create a virtual environment
python -m venv .venv
source .venv/bin/activate      # macOS / Linux
.venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install streamlit nltk scikit-learn numpy

# 4. Run the Streamlit app
streamlit run chatbot_app.py
```

The app will open automatically at `http://localhost:8501`.

### CLI Mode

You can also run the chatbot directly in the terminal without Streamlit:

```bash
python chatbot_app.py
```

Type `quit` to exit the terminal session.

---

## 🧠 How It Works

```
User Input
    │
    ▼
Preprocessing
  • Lowercase
  • Remove punctuation
  • Tokenize (NLTK)
  • Remove stopwords
  • Lemmatize (WordNetLemmatizer)
    │
    ▼
Vectorization
  • Build shared vocabulary across all intents
  • Convert user message → bag-of-words vector
  • Convert each pattern → bag-of-words vector
    │
    ▼
Intent Matching (Cosine Similarity)
  • Compare user vector against every pattern vector
  • Track best-matching intent and per-intent scores
  • Threshold: score ≥ 0.3 → matched intent
             score < 0.3 → fallback response
    │
    ▼
Response Selection
  • Pick a random response from the matched intent's pool
  • Return response + all intent confidence scores
```

---

## 💬 Supported Intents

| Intent | Example Patterns |
|---|---|
| `greeting` | "hello", "hi", "good morning" |
| `goodbye` | "bye", "see you", "take care" |
| `thanks` | "thank you", "much appreciated" |
| `sad` | "i am sad", "i feel lonely" |
| `happy` | "i am happy", "feeling good" |
| `angry` | "i am frustrated", "i hate this" |
| `anxious` | "i am stressed", "i am scared" |
| `study` | "help me study", "give me study tips" |
| `motivation` | "motivate me", "i want to give up" |
| `name` | "what is your name", "who are you" |
| `age` | "how old are you" |
| `time` | "what time is it" |
| `joke` | "tell me a joke" |
| `weather` | "what is the weather" |
| `help` | "can you help me", "i need help" |
| `fallback` | *(any unrecognised input)* |

---

## ✨ Features

- **Streamlit web UI** with a clean chat interface and persistent message history
- **Real-time intent scores** shown in the sidebar — see confidence values for every intent after each message
- **CLI mode** — run and test the bot directly in the terminal
- **Easily extensible** — add new intents by appending entries to the `intents` list in `chatbot_app.py`
- **Fallback handling** — graceful responses when no intent matches

---

## 🗂️ Project Structure

```
chatbot/
├── chatbot_app.py      # Main app: intents, NLP logic, Streamlit UI
├── chatbot_nltk.py     # Standalone NLTK-based chatbot module
├── streamlish.py       # Streamlit helper / entry point
```

---

## 🛠️ Tech Stack

| Layer | Library | Version |
|---|---|---|
| NLP | `nltk` | ≥ 3.8 |
| ML / Similarity | `scikit-learn` | ≥ 1.3 |
| Numerics | `numpy` | ≥ 1.24 |
| Web UI | `streamlit` | ≥ 1.30 |

---

## 🔧 Extending the Bot

To add a new intent, open `chatbot_app.py` and append a new entry to the `intents` list:

```python
{
    "tag": "my_new_intent",
    "patterns": [
        "example phrase one",
        "another example phrase"
    ],
    "responses": [
        "Response option A",
        "Response option B"
    ]
}
```

No retraining is required — the bot rebuilds its vocabulary dynamically at runtime.

---

## 🙌 Acknowledgements

- [NLTK](https://www.nltk.org/) — Natural Language Toolkit
- [scikit-learn](https://scikit-learn.org/) — Machine learning in Python
- [Streamlit](https://streamlit.io/) — Fast web app framework for ML projects
