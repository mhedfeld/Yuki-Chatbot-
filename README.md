# AI Chatbot with Neural Network and NLP

## Project Overview
This project implements an AI Chatbot using Python, Neural Networks, and Natural Language Processing (NLP) techniques. The chatbot is designed to understand user intents, provide relevant responses, and continuously improve through user feedback and periodic model updates.

## Problem Solved
This AI Chatbot addresses the challenge of providing automated, intelligent responses to user queries in a conversational interface. It uses advanced AI techniques to understand user intent and generate appropriate responses, making it suitable for various applications such as customer support, information retrieval, and interactive user assistance.

## AI Technology Applied
The project utilizes the following AI technologies:
1. Natural Language Processing (NLP) for text preprocessing and understanding
2. Neural Networks for intent classification
3. Transfer Learning techniques (TF-IDF and LSA) for feature extraction
4. Continuous Learning through feedback collection and model updates

## Algorithm Applied
The main algorithm used in this project is a Deep Neural Network for intent classification. The model architecture includes:
- Input layer based on TF-IDF and LSA transformed features
- Multiple dense layers with ReLU activation
- Dropout layers for regularization
- Output layer with softmax activation for multi-class classification

The model is trained using the Adam optimizer and sparse categorical crossentropy loss function.

## Repository Structure

project_root/
│
├── data/
│   └── your_custom_data.json
│
├── images/
│   ├── user_icon.png
│   └── bot_icon.png
│
├── models/
│   ├── intent_classifier.pkl
│   ├── tfidf_vectorizer.pkl
│   └── lsa.pkl
│
├── src/
│   ├── data_preparation.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── model.py
│   ├── hyperparameter_tuning.py
│   └── modern_chatbot_ui.py
│
├── custom_widgets.py
├── evaluate_and_retrain.py
├── feedback_handler.py
├── main.py
├── model_updater.py
├── requirements.txt
└── README.md

## How to Start the System
1. Ensure you have Python 3.12 installed on your system.
2. Clone this repository to your local machine.
3. Navigate to the project root directory in your terminal.
4. Install the required dependencies:
pip install -r requirements.txt
5. Run the main script to start the chatbot:


## Additional Information
- The chatbot uses a modern GUI implemented with customtkinter.
- User feedback is collected and stored in a SQLite database.
- The model is periodically retrained using collected feedback and new data.
- Hyperparameter tuning is performed to optimize the model's performance.