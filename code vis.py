import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
import nltk
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Download NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Preprocessing function
def preprocess_text(text):
    """
    Basic text preprocessing
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation and numbers
    import re
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    return text

# Create sample dataset
def create_sample_dataset():
    """
    Create a synthetic dataset for fake news detection
    """
    fake_news = [
        "Government conspiracy revealed shocking truth",
        "Miracle cure discovered overnight",
        "Aliens planning secret invasion",
        "Shocking celebrity scandal exposed"
    ]
    
    real_news = [
        "Economic growth shows positive trends",
        "Scientific research advances medical understanding",
        "Local community initiatives make progress",
        "International diplomacy continues dialogue"
    ]
    
    # Create DataFrame
    df = pd.DataFrame({
        'text': fake_news + real_news,
        'label': [0]*len(fake_news) + [1]*len(real_news)
    })
    
    return df

# Main function to train and evaluate the model
def train_fake_news_detector():
    # Create dataset
    df = create_sample_dataset()
    
    # Preprocess text
    df['processed_text'] = df['text'].apply(preprocess_text)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        df['processed_text'], df['label'], test_size=0.2, random_state=42
    )
    
    # Vectorize the text
    vectorizer = TfidfVectorizer(stop_words='english')
    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)
    
    # Train logistic regression model
    model = LogisticRegression()
    model.fit(X_train_vectorized, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_vectorized)
    
    # Evaluate the model
    print("Model Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()
    
    # Precision, Recall, F1-Score
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    metrics = ['Precision', 'Recall', 'F1 Score']
    scores = [precision, recall, f1]

    plt.figure(figsize=(6, 4))
    sns.barplot(x=metrics, y=scores, palette='viridis')
    plt.title('Precision, Recall, and F1-Score')
    plt.show()
    
    # Word Cloud for Fake and Real News
    fake_text = ' '.join(df[df['label'] == 0]['processed_text'])
    real_text = ' '.join(df[df['label'] == 1]['processed_text'])

    fake_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(fake_text)
    real_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(real_text)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(fake_wordcloud, interpolation='bilinear')
    plt.title('Fake News Word Cloud')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(real_wordcloud, interpolation='bilinear')
    plt.title('Real News Word Cloud')
    plt.axis('off')

    plt.show()

    return model, vectorizer

# Run the detector
if __name__ == "__main__":
    model, vectorizer = train_fake_news_detector()
