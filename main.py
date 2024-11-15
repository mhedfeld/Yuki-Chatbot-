import numpy as np
import pandas as pd
import pickle
import time
import string
import nltk
import random
import threading
import os
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from src.data_preparation import load_and_prepare_data
from src.preprocessing import preprocess_dataframe
from src.feature_engineering import create_tfidf_vectorizer
from src.model import train_intent_classifier, predict_intent
from src.hyperparameter_tuning import tune_hyperparameters
from src.modern_chatbot_ui import ModernChatbotUI
from model_updater import ModelUpdater
from feedback_handler import FeedbackHandler

class DecisionNode:
    def __init__(self, question, yes_node, no_node, response=None, intent=None):
        self.question = question
        self.yes_node = yes_node
        self.no_node = no_node
        self.response = response
        self.intent = intent

def create_decision_tree():
    # Create a simple decision tree for demonstration
    farewell = DecisionNode(None, None, None, "Thank you for chatting with us. Have a great day!", "farewell")
    support = DecisionNode(None, None, None, "I'm here to help. What issue are you facing?", "support")
    order = DecisionNode(None, None, None, "Great! Let's start your order. What would you like to order?", "order")
    
    main_menu = DecisionNode(
        "Would you like to place an order?",
        order,
        support,
        "How can I help you today?",
        "main_menu"
    )
    root = DecisionNode(
        "Is this your first message?",
        main_menu,
        farewell,
        "Welcome! How can I assist you?",
        "greeting"
    )
    return root

decision_tree = create_decision_tree()

def traverse_decision_tree(node, user_input, intent, depth=0):
    if depth > 5 or node is None:  # Limit depth to prevent excessive recursion
        return None

    if node.intent == intent:
        if node.response:
            return node.response
        elif node.question:
            if "yes" in user_input.lower():
                return traverse_decision_tree(node.yes_node, user_input, intent, depth + 1)
            else:
                return traverse_decision_tree(node.no_node, user_input, intent, depth + 1)

    # If intent doesn't match, try both branches
    yes_response = traverse_decision_tree(node.yes_node, user_input, intent, depth + 1)
    if yes_response:
        return yes_response
    return traverse_decision_tree(node.no_node, user_input, intent, depth + 1)

def preprocess_text(text):
    # Tokenize the text
    tokens = word_tokenize(text.lower())
    
    # Remove punctuation and numbers
    tokens = [token for token in tokens if token not in string.punctuation and not token.isdigit()]
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    # Lemmatize the tokens
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    # Join the tokens back into a string
    return ' '.join(tokens)

def main():
    global df, model, label_encoder, tfidf_vectorizer, lsa

    print("Initializing chatbot...")
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)

    print("Loading and preparing data...")
    df = load_and_prepare_data('data/your_custom_data.json')

    print("Preprocessing data...")
    df['preprocessed_query'] = df['query'].apply(preprocess_text)

    print("Performing feature engineering...")
    tfidf_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['preprocessed_query'])
    
    # Get the number of features in the TF-IDF matrix
    n_features = tfidf_matrix.shape[1]
    
    # Set the number of components for LSA
    n_components = min(n_features - 1, 150)  # Use 150 or fewer components
    
    lsa = TruncatedSVD(n_components=n_components, random_state=42)
    X = lsa.fit_transform(tfidf_matrix)
    y = df['intent']
    input_shape = (X.shape[1],)
    num_classes = len(np.unique(y))

    print("Tuning hyperparameters...")
    best_params = tune_hyperparameters(X, y, input_shape=input_shape, num_classes=num_classes)
    print("Hyperparameter tuning complete")

    print("Training model...")
    model, label_encoder, history = train_intent_classifier(X, y, **best_params)
    print("Model training complete")

    print("Saving models...")
    with open('models/tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(tfidf_vectorizer, f)
    with open('models/lsa.pkl', 'wb') as f:
        pickle.dump(lsa, f)
    with open('models/intent_classifier.pkl', 'wb') as f:
        pickle.dump((model, label_encoder), f)

    print("Initializing feedback handler and model updater...")
    feedback_db_path = 'feedback.db'
    feedback_handler = FeedbackHandler(feedback_db_path)
    model_updater = ModelUpdater(
        model_path='models/intent_classifier.pkl',
        data_path='data/your_custom_data.json',
        vectorizer_path='models/tfidf_vectorizer.pkl',
        lsa_path='models/lsa.pkl',
        feedback_handler=feedback_handler
    )

    # Comment out the thread start for now
    # updater_thread = threading.Thread(target=model_updater.schedule_updates)
    # updater_thread.start()

    print("Feedback handler and model updater initialized.")

    def response_generator(query):
        preprocessed_query = preprocess_text(query)
        try:
            intent, confidence = predict_intent(model, label_encoder, tfidf_vectorizer, lsa, preprocessed_query, confidence_threshold=0.1)
        except Exception as e:
            print(f"Error in intent prediction: {e}")
            intent = "unknown"
            confidence = 0

        # Always try to get a response from the decision tree first
        tree_response = traverse_decision_tree(decision_tree, query, intent)
        if tree_response:
            return tree_response

        # If no tree response, use intent-based response or fallback
        if intent != "unknown" and confidence >= 0.1:
            possible_responses = df[df['intent'] == intent]['response'].tolist()
            if possible_responses:
                return random.choice(possible_responses)

        # Fallback responses
        fallback_responses = [
            "I'm not sure I understood that correctly. Could you please rephrase your question?",
            "I'm having trouble understanding. Can you provide more details?",
            "I'm sorry, but I don't have a specific answer for that. Is there something else I can help you with?",
            "I'm still learning and may not have the answer to that. Can you try asking in a different way?",
            "I apologize, but I'm not able to provide a relevant answer at the moment. Is there another topic you'd like to discuss?"
        ]
        return random.choice(fallback_responses)

    def feedback_handler_func(user_message, bot_message, is_helpful):
        feedback_handler.add_feedback(user_message, bot_message, is_helpful)
        model_updater.process_feedback()

    print("Initialization complete. Launching GUI...")
    app = ModernChatbotUI(response_generator, feedback_handler_func)
    app.mainloop()

if __name__ == "__main__":
    main()