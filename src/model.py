import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
import pickle

def create_model(neurons1=256, neurons2=128, neurons3=64, dropout_rate=0.5, learning_rate=0.001, l2_reg=0.01, input_shape=None, num_classes=None):
    if input_shape is None or num_classes is None:
        raise ValueError("input_shape and num_classes must be provided")
    
    model = Sequential([
        Input(shape=input_shape),
        Dense(neurons1, activation='relu', kernel_regularizer=l2(l2_reg)),
        Dropout(dropout_rate),
        Dense(neurons2, activation='relu', kernel_regularizer=l2(l2_reg)),
        Dropout(dropout_rate),
        Dense(neurons3, activation='relu', kernel_regularizer=l2(l2_reg)),
        Dropout(dropout_rate),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def train_intent_classifier(X, y, **kwargs):
    # Encode the intents
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    # Remove the 'model__' prefix from kwargs
    model_kwargs = {k.replace('model__', ''): v for k, v in kwargs.items() if k.startswith('model__')}
    
    # Add input_shape and num_classes to model_kwargs
    model_kwargs['input_shape'] = (X.shape[1],)
    model_kwargs['num_classes'] = len(np.unique(y_encoded))

    # Create the model using the best parameters
    model = create_model(**model_kwargs)

    # Train the model
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    history = model.fit(X_train, y_train, validation_split=0.2, verbose=1, 
                        batch_size=kwargs.get('batch_size', 32),
                        epochs=kwargs.get('epochs', 100),
                        callbacks=[early_stopping])

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test accuracy: {accuracy:.4f}")

    return model, label_encoder, history

def predict_intent(model, label_encoder, tfidf_vectorizer, lsa, query, confidence_threshold=0.35):
    query_vector = tfidf_vectorizer.transform([query])
    query_vector_lsa = lsa.transform(query_vector)
    intent_probabilities = model.predict(query_vector_lsa)[0]
    predicted_intent_index = np.argmax(intent_probabilities)
    confidence = intent_probabilities[predicted_intent_index]
    
    print("Debug - Query:", query)
    print("Debug - Intent probabilities:", intent_probabilities)
    print("Debug - Predicted intent index:", predicted_intent_index)
    print("Debug - Confidence:", confidence)
    print("Debug - Confidence threshold:", confidence_threshold)
    
    if confidence >= confidence_threshold:
        predicted_intent = label_encoder.inverse_transform([predicted_intent_index])[0]
        print("Debug - Predicted intent:", predicted_intent)
        return predicted_intent, confidence
    else:
        print("Debug - Intent: unknown")
        return "unknown", confidence