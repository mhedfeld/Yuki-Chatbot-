import schedule
import time
import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from src.data_preparation import load_and_prepare_data
from src.preprocessing import preprocess_text
from src.feature_engineering import create_tfidf_vectorizer

class ModelUpdater:
    def __init__(self, model_path, data_path, vectorizer_path, lsa_path, feedback_handler):
        self.model_path = model_path
        self.data_path = data_path
        self.vectorizer_path = vectorizer_path
        self.lsa_path = lsa_path
        self.feedback_handler = feedback_handler

    def process_feedback(self):
        # This method will be called when feedback is received
        print("Feedback received. Model update may be needed.")
        

    def update_training_data(self):
        # Load existing data
        df = load_and_prepare_data(self.data_path)

        # Get helpful feedback
        helpful_feedback = self.feedback_handler.get_helpful_feedback()

        # Process and add helpful feedback to training data
        new_data = []
        for user_input, bot_response in helpful_feedback:
            new_data.append({
                'query': user_input,
                'response': bot_response,
                'intent': 'new_intent'  
            })
        
        # Concatenate new data with existing data
        if new_data:
            new_df = pd.DataFrame(new_data)
            df = pd.concat([df, new_df], ignore_index=True)

        # Save updated data
        df.to_json(self.data_path, orient='records')

        return df

    def retrain_model(self):
        df = self.update_training_data()
        
        # Preprocess the data
        df['preprocessed_query'] = df['query'].apply(preprocess_text)

        # Feature engineering
        tfidf_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        tfidf_matrix = tfidf_vectorizer.fit_transform(df['preprocessed_query'])
        
        # Get the number of features in the TF-IDF matrix
        n_features = tfidf_matrix.shape[1]
        
        # Set the number of components for LSA
        n_components = min(n_features - 1, 300)  # Use 300 or fewer components
        
        lsa = TruncatedSVD(n_components=n_components, random_state=42)
        X = lsa.fit_transform(tfidf_matrix)
        
        # Prepare target variable
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(df['intent'])

        # Create and train the model
        model = self.create_model(input_shape=(X.shape[1],), num_classes=len(np.unique(y)))
        model.fit(X, y, epochs=100, batch_size=32, validation_split=0.2, verbose=1)

        # Save the new model and related objects
        with open(self.model_path, 'wb') as f:
            pickle.dump((model, label_encoder), f)
        
        with open(self.vectorizer_path, 'wb') as f:
            pickle.dump(tfidf_vectorizer, f)
        
        with open(self.lsa_path, 'wb') as f:
            pickle.dump(lsa, f)

        print("Model retrained and saved successfully.")

    def create_model(self, input_shape, num_classes):
        model = Sequential([
            Input(shape=input_shape),
            Dense(256, activation='relu'),
            Dropout(0.5),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(64, activation='relu'),
            Dropout(0.5),
            Dense(num_classes, activation='softmax')
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

    def schedule_updates(self):
        schedule.every().day.at("02:00").do(self.retrain_model)  # Retrain daily at 2 AM

        while True:
            schedule.run_pending()
            time.sleep(1)

if __name__ == "__main__":
    # This is for testing purposes only
    from feedback_handler import FeedbackHandler

    feedback_handler = FeedbackHandler('feedback.db')
    model_updater = ModelUpdater(
        model_path='models/intent_classifier.pkl',
        data_path='data/your_custom_data.json',
        vectorizer_path='models/tfidf_vectorizer.pkl',
        lsa_path='models/lsa.pkl',
        feedback_handler=feedback_handler
    )

    # Simulate some feedback
    feedback_handler.add_feedback("How do I track my order?", "You can track your order by...", True)
    feedback_handler.add_feedback("What's your return policy?", "Our return policy allows...", True)

    # Run the model update (for testing purposes)
    model_updater.retrain_model()

    # model_updater.schedule_updates()
