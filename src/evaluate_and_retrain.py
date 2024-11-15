import pandas as pd
import pickle
from src.data_preparation import load_and_prepare_data
from src.preprocessing import preprocess_dataframe
from src.feature_engineering import create_tfidf_vectorizer
from src.model import train_intent_classifier, predict_intent

def evaluate_model(model, label_encoder, tfidf_vectorizer, lsa, test_data):
    correct = 0
    total = len(test_data)
    
    for item in test_data:
        query = item['query']
        true_intent = item['intent']
        predicted_intent, _ = predict_intent(model, label_encoder, tfidf_vectorizer, lsa, query)
        
        if predicted_intent == true_intent:
            correct += 1
    
    accuracy = correct / total
    print(f"Model Accuracy: {accuracy:.4f}")
    return accuracy

def retrain_model(threshold=0.8):
    # Load and prepare new data
    df = load_and_prepare_data('data/new_data.json')
    df = preprocess_dataframe(df)
    
    # Load existing models
    with open('models/tfidf_vectorizer.pkl', 'rb') as f:
        tfidf_vectorizer = pickle.load(f)
    
    with open('models/lsa.pkl', 'rb') as f:
        lsa = pickle.load(f)
    
    with open('models/intent_classifier.pkl', 'rb') as f:
        model, label_encoder = pickle.load(f)
    
    # Prepare new data
    new_tfidf_matrix = tfidf_vectorizer.transform(df['preprocessed_query'])
    new_lsa_matrix = lsa.transform(new_tfidf_matrix)
    
    # Evaluate current model on new data
    accuracy = evaluate_model(model, label_encoder, tfidf_vectorizer, lsa, df.to_dict('records'))
    
    if accuracy < threshold:
        print(f"Accuracy {accuracy:.4f} below threshold {threshold}. Retraining model...")
        
        # Combine old and new data
        old_df = pd.read_csv('data/training_data.csv')
        combined_df = pd.concat([old_df, df])
        
        # Retrain model
        combined_tfidf_matrix = tfidf_vectorizer.transform(combined_df['preprocessed_query'])
        combined_lsa_matrix = lsa.transform(combined_tfidf_matrix)
        
        X = combined_lsa_matrix
        y = combined_df['intent']
        
        new_model, new_label_encoder = train_intent_classifier(X, y)
        
        # Save new model
        with open('models/intent_classifier.pkl', 'wb') as f:
            pickle.dump((new_model, new_label_encoder), f)
        
        print("Model retrained and saved.")
    else:
        print(f"Model accuracy {accuracy:.4f} above threshold. No retraining needed.")

if __name__ == "__main__":
    retrain_model()