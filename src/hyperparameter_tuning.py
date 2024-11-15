from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam

def create_model(neurons1=256, neurons2=128, neurons3=64, dropout_rate=0.5, learning_rate=0.001, input_shape=None, num_classes=None):
    if input_shape is None or num_classes is None:
        raise ValueError("input_shape and num_classes must be provided")
    
    model = Sequential([
        Input(shape=input_shape),
        Dense(neurons1, activation='relu'),
        Dropout(dropout_rate),
        Dense(neurons2, activation='relu'),
        Dropout(dropout_rate),
        Dense(neurons3, activation='relu'),
        Dropout(dropout_rate),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def tune_hyperparameters(X, y, input_shape, num_classes):
    model = KerasClassifier(model=create_model, verbose=0)
    
    param_dist = {
        'model__neurons1': [64, 128, 256, 512],
        'model__neurons2': [32, 64, 128, 256],
        'model__neurons3': [16, 32, 64, 128],
        'model__dropout_rate': [0.2, 0.3, 0.4, 0.5],
        'model__learning_rate': [0.0001, 0.001, 0.01],
        'model__input_shape': [input_shape],
        'model__num_classes': [num_classes],
        'batch_size': [16, 32, 64],
        'epochs': [50, 100, 150, 200]
    }
    
    random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, n_iter=30, cv=3, verbose=2, n_jobs=-1)
    random_search_result = random_search.fit(X, y)
    
    print("Best: %f using %s" % (random_search_result.best_score_, random_search_result.best_params_))
    return random_search_result.best_params_