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

project_root/<br>
│<br>
├── data/<br>
│   └── your_custom_data.json<br>
│<br>
├── images/<br>
│   ├── user_icon.png<br>
│   └── bot_icon.png<br>
│<br>
├── models/<br>
│   ├── intent_classifier.pkl<br>
│   ├── tfidf_vectorizer.pkl<br>
│   └── lsa.pkl<br>
│<br>
├── src/<br>
│   ├── data_preparation.py<br>
│   ├── preprocessing.py<br>
│   ├── feature_engineering.py<br>
│   ├── model.py<br>
│   ├── hyperparameter_tuning.py<br>
│   └── modern_chatbot_ui.py<br>
│<br>
├── custom_widgets.py<br>
├── evaluate_and_retrain.py<br>
├── feedback_handler.py<br>
├── main.py<br>
├── model_updater.py<br>
├── requirements.txt<br>
└── README.md<br>

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

- The Model Scales in performance with increased usage.
- Unfortunately during my tesing, Sometimes after updating the model with the user feedback, would sometimes cause unexpected decreses in acuraccy, due to faulty intent classification. I was not able to adress this Issue adequately yet.

- The Other Gui file, located in the repo is a legacy version of my initial UI. 
