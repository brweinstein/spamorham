import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.sparse as sp

def load_data(path: str = "data/spam.csv"):
    try:
        df = pd.read_csv(path, encoding='utf-8') 
    except UnicodeDecodeError: 
        df = pd.read_csv(path, encoding='latin1') 

    df = df[['v1', 'v2']].dropna() 
    df = df.rename(columns={'v1': 'label', 'v2': 'text'}) 
    df['label'] = df['label'].map(lambda x: 0 if x == 'ham' else 1)

    return df

def extract_features(df, max_features=2000):
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
    X_tfidf = vectorizer.fit_transform(df['text'])

    df['text_length'] = df['text'].apply(len)
    df['num_exclam'] = df['text'].apply(lambda x: x.count('!'))
    df['num_digits'] = df['text'].apply(lambda x: sum(c.isdigit() for c in x))

    X_numeric = df[['text_length', 'num_exclam', 'num_digits']].values
    X = sp.hstack([X_tfidf, X_numeric])

    return X, df['label'], vectorizer

