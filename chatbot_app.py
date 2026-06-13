import streamlit as st
import nltk
import numpy as np
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity
import random


nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')



intents = [
    # ─── GREETINGS ───────────────────────────────────────
    {
        "tag": "greeting",
        "patterns": [
            "hello", "hi", "hey", "good morning",
            "good afternoon", "good evening",
            "hi there", "whats up", "howdy"
        ],
        "responses": [
            "Hello! How can I help you today?",
            "Hi there! What can I do for you?",
            "Hey! Great to see you! How can I assist?"
        ]
    },

    # ─── GOODBYE ─────────────────────────────────────────
    {
        "tag": "goodbye",
        "patterns": [
            "bye", "goodbye", "see you", "take care",
            "talk later", "i am leaving", "catch you later"
        ],
        "responses": [
            "Goodbye! Have a great day!",
            "See you later! Take care!",
            "Bye! Feel free to come back anytime!"
        ]
    },

    # ─── THANKS ──────────────────────────────────────────
    {
        "tag": "thanks",
        "patterns": [
            "thank you", "thanks", "thank you so much",
            "thanks a lot", "much appreciated", "that was helpful"
        ],
        "responses": [
            "You're welcome!",
            "Happy to help!",
            "Anytime! Let me know if you need anything else!"
        ]
    },

    # ─── EMOTIONS: SAD ───────────────────────────────────
    {
        "tag": "sad",
        "patterns": [
            "i feel bad", "i am sad", "i am not good",
            "not feeling well", "i am feeling low",
            "i am depressed", "i feel lonely",
            "i am upset", "i am broken"
        ],
        "responses": [
            "I'm sorry to hear that. It's okay to feel this way sometimes.",
            "Everything will be fine, hang in there!",
            "I'm here for you. Want to talk about it?"
        ]
    },

    # ─── EMOTIONS: HAPPY ─────────────────────────────────
    {
        "tag": "happy",
        "patterns": [
            "i am happy", "i feel great", "i am excited",
            "i am doing well", "feeling good", "i am awesome",
            "i am on top of the world", "life is good"
        ],
        "responses": [
            "That's wonderful to hear!",
            "Amazing! Keep that positive energy going!",
            "So glad to hear that! You deserve it!"
        ]
    },

    # ─── EMOTIONS: ANGRY ─────────────────────────────────
    {
        "tag": "angry",
        "patterns": [
            "i am angry", "i am frustrated", "this is annoying",
            "i am mad", "i hate this", "i am furious",
            "i cant take this anymore"
        ],
        "responses": [
            "I understand your frustration. Take a deep breath!",
            "It's okay to feel angry sometimes. Want to talk about it?",
            "I'm sorry you're going through this. How can I help?"
        ]
    },

    # ─── EMOTIONS: ANXIOUS ───────────────────────────────
    {
        "tag": "anxious",
        "patterns": [
            "i am anxious", "i am worried", "i am nervous",
            "i am stressed", "i am overwhelmed",
            "i cant do this", "i am scared"
        ],
        "responses": [
            "Take a deep breath, you've got this!",
            "It's okay to feel nervous. Break it down into small steps!",
            "I believe in you! One step at a time."
        ]
    },

    # ─── STUDY RELATED ───────────────────────────────────
    {
        "tag": "study",
        "patterns": [
            "how can i study", "help me study",
            "i need to focus", "i want to learn",
            "how to improve my grades", "i cant concentrate",
            "give me study tips"
        ],
        "responses": [
            "Try the Pomodoro technique — 25 mins study, 5 mins break!",
            "Break your syllabus into small topics and tackle one at a time!",
            "Consistency is key! Even 1 hour daily makes a big difference."
        ]
    },

    # ─── MOTIVATION ──────────────────────────────────────
    {
        "tag": "motivation",
        "patterns": [
            "motivate me", "i need motivation",
            "i want to give up", "i feel like quitting",
            "i am not good enough", "i cant do it",
            "inspire me"
        ],
        "responses": [
            "Believe in yourself! Every expert was once a beginner.",
            "Don't give up! The hardest paths lead to the greatest destinations.",
            "You are capable of amazing things. Keep going!"
        ]
    },

    # ─── SMALL TALK: JOKES ───────────────────────────────
    {
        "tag": "joke",
        "patterns": [
            "tell me a joke", "say something funny",
            "make me laugh", "i need a joke"
        ],
        "responses": [
            "Why do programmers prefer dark mode? Because light attracts bugs! 🐛",
            "Why was the math book sad? It had too many problems! 😄",
            "I told my computer I needed a break. Now it won't stop sending me Kit-Kat ads! 😂"
        ]
    },

    # ─── SMALL TALK: HOW ARE YOU ─────────────────────────
    {
        "tag": "howareyou",
        "patterns": [
            "how are you", "how are you doing",
            "are you okay", "how do you feel",
            "what about you"
        ],
        "responses": [
            "I am just a bot, but I'm doing great! How about you?",
            "Running perfectly! Thanks for asking. How can I help?",
            "All good on my end! What can I do for you?"
        ]
    },

    # ─── SMALL TALK: TIME ────────────────────────────────
    {
        "tag": "time",
        "patterns": [
            "what time is it", "tell me the time",
            "current time", "what is the time now"
        ],
        "responses": [
            "I don't have a clock, but your device does! Check the top of your screen 😊",
            "I can't check the time, but I'm sure you can find it nearby!"
        ]
    },

    # ─── HELP ────────────────────────────────────────────
    {
        "tag": "help",
        "patterns": [
            "help me", "i need help", "can you help",
            "i have a problem", "i am stuck", "i need assistance"
        ],
        "responses": [
            "Sure! I'm here to help. What do you need?",
            "Of course! Tell me what's going on.",
            "I'm ready to help! What's the issue?"
        ]
    },

    # ─── GUIDE ───────────────────────────────────────────
    {
        "tag": "guide",
        "patterns": [
            "how can i do this", "guide me",
            "explain this to me", "i want to understand",
            "walk me through", "show me how"
        ],
        "responses": [
            "Sure! Let's work through this step by step.",
            "Happy to guide you! Where would you like to start?",
            "Let's figure this out together!"
        ]
    },

    # ─── FALLBACK ────────────────────────────────────────
    {
        "tag": "fallback",
        "patterns": [],
        "responses": [
            "Sorry, I didn't understand that. Can you rephrase?",
            "Hmm, I'm not sure about that. Can you try again?",
            "I didn't quite catch that. Could you say it differently?"
        ]
    }
]



lemmatizer = WordNetLemmatizer()

stop_words = set(stopwords.words('english'))
negations = {"not", "no", "never", "neither", "nor", "none", "how", "are", "you"}
custom_stopwords = stop_words - negations

def preprocess(sentence):
    # Step 1: lowercase
    sentence = sentence.lower()
    
    # Step 2: tokenize
    tokens = word_tokenize(sentence)
    
    # Step 3: remove punctuation
    tokens = [word for word in tokens if word not in string.punctuation ]
    
    # Step 4: remove stopwords
    tokens = [word for word in tokens if word not in custom_stopwords]
    
    # Step 5: lemmatize (treat all as verbs)
    tokens = [lemmatizer.lemmatize(word, pos='v') for word in tokens]
    
    return tokens


vocabulary = []

for intent in intents:                          # loop through intents list
    for pattern in intent["patterns"]:             # loop through patterns
        words = preprocess(pattern)             # preprocess the pattern
        for word in words:                    # loop through words
            if word not in vocabulary:       # avoid duplicates
                vocabulary.append(word)      # add to vocabulary

print("Total vocabulary size:", len(vocabulary))
print("Vocabulary:", vocabulary)


def vectorize(words):
    vector = []
    for word in vocabulary:
        if word in words:        # word exists in words list
            vector.append(1)
        else:
            vector.append(0)
    return vector




def get_response(user_input):
    user_words = preprocess(user_input)
    user_vector = vectorize(user_words)
    
    best_intent = None
    best_score = 0.0
    intent_scores = {}          # ← NEW: track scores for all intents
    
    for intent in intents:
        if intent["tag"] == "fallback":
            continue
            
        for pattern in intent["patterns"]:
            pattern_words = preprocess(pattern)
            pattern_vector = vectorize(pattern_words)
            
            score = cosine_similarity(
                [user_vector],
                [pattern_vector]
            )[0][0]
            
            if score > best_score:
                best_score = score
                best_intent = intent["tag"]
            
            # ← NEW: keep highest score per intent
            if intent["tag"] not in intent_scores:
                intent_scores[intent["tag"]] = round(float(score), 2)
            else:
                if score > intent_scores[intent["tag"]]:
                    intent_scores[intent["tag"]] = round(float(score), 2)
    
    if best_score < 0.3:
        best_intent = "fallback"
    
    for intent in intents:
        if intent["tag"] == best_intent:
            return random.choice(intent["responses"]), intent_scores  # ← NEW
        

def chat():
    print("Hi! I am your chatbot. Type 'quit' to exit.")
    
    while True:                              # run forever
        user_input = input("You: ")          # get user input
        
        if user_input.lower()=="quit":                             # check if user typed 'quit'
            print("Bot: Goodbye! Take care!")
            break                             # stop the loop
        
        response = get_response(user_input)                      # get response from our function
        print("Bot:", response)                  # print it



# ─── STREAMLIT UI ───────────────────────────

st.title("🤖 My Chatbot")
st.caption("Ask me anything!")

# initialize session state ONCE
if "messages" not in st.session_state:
    st.session_state.messages = []    # empty list → stores chat history

if "scores" not in st.session_state:
    st.session_state.scores = {}      # empty dict → stores intent scores

# ─── SIDEBAR ────────────────────────────────
with st.sidebar:
    st.title("🎯 Intent Scores")
    st.caption("Shows how well your input matched each intent")
    
    if st.session_state.scores:          # only show if scores exist
        for intent, score in st.session_state.scores.items():
            st.write(f"{intent}: {score:.2f}")
    else:
        st.write("Send a message to see scores!")

# ─── CHAT HISTORY ───────────────────────────
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# ─── CHAT INPUT ─────────────────────────────
user_input = st.chat_input("Type your message here...")

if user_input:
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })
    
    response, scores = get_response(user_input)    # ← unpack both!
    st.session_state.scores = scores               # ← save scores!
    
    st.session_state.messages.append({
        "role": "assistant",
        "content": response
    })
    
    st.rerun()
# start the chat
chat()

