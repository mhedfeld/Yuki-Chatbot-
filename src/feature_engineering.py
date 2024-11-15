from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import pickle
import pandas as pd

def create_tfidf_vectorizer(df):
    tfidf_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['preprocessed_query'])
    
    # Apply LSA to reduce dimensionality
    lsa = TruncatedSVD(n_components=100, random_state=42)
    lsa_matrix = lsa.fit_transform(tfidf_matrix)
    
    return tfidf_vectorizer, lsa, lsa_matrix

if __name__ == "__main__":
    # Load preprocessed data
    df = pd.read_csv('data/preprocessed_data.csv')
    
    # Create TF-IDF vectorizer and LSA
    vectorizer, lsa, _ = create_tfidf_vectorizer(df)
    
    # Save TF-IDF vectorizer
    with open('models/tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    
    # Save LSA model
    with open('models/lsa.pkl', 'wb') as f:
        pickle.dump(lsa, f)
    
    print("Feature engineering complete.")