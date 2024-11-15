from sklearn.metrics.pairwise import cosine_similarity
import pickle
import pandas as pd

class ResponseGenerator:
    def __init__(self, tfidf_vectorizer, tfidf_matrix_query, df):
        self.tfidf_vectorizer = tfidf_vectorizer
        self.tfidf_matrix_query = tfidf_matrix_query
        self.df = df

    def generate_response(self, query):
        query_vector = self.tfidf_vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.tfidf_matrix_query)
        most_similar_idx = similarities.argmax()
        return self.df.iloc[most_similar_idx]['response']

if __name__ == "__main__":
    df = pd.read_csv('data/preprocessed_data.csv')
    
    with open('models/tfidf_vectorizer.pkl', 'rb') as f:
        tfidf_vectorizer = pickle.load(f)
    
    tfidf_matrix_query = tfidf_vectorizer.transform(df['query'])
    
    response_generator = ResponseGenerator(tfidf_vectorizer, tfidf_matrix_query, df)
    
    test_query = "How can I track my order?"
    response = response_generator.generate_response(test_query)
    print(f"Query: {test_query}")
    print(f"Response: {response}")