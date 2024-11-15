import pandas as pd
import json

def load_and_prepare_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Check if 'data' key exists in the JSON
    if 'data' in data:
        df = pd.DataFrame(data['data'])
    else:
        df = pd.DataFrame(data)
    
    # Check if 'query' and 'response' columns exist
    if 'query' not in df.columns or 'response' not in df.columns:
        print("Error: 'query' or 'response' column not found in the data.")
        print("Available columns:", df.columns.tolist())
        raise KeyError("Required columns not found in the data")
    
    # Clean and preprocess the data
    df['query'] = df['query'].str.strip().str.lower()
    df['response'] = df['response'].str.strip()
    
    # Remove duplicates and null values
    df = df.dropna().drop_duplicates()
    
    return df

if __name__ == "__main__":
    df = load_and_prepare_data('data/your_custom_data.json')
    print(f"Cleaned data shape: {df.shape}")
    print("Columns:", df.columns.tolist())
    print("\nFirst few rows:")
    print(df.head())