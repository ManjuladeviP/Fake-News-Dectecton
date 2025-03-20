# Fake News Detection System

This Python application detects fake news using Natural Language Processing (NLP) and Machine Learning techniques. It utilizes a logistic regression model trained on a dataset of fake and real news headlines. The application processes text data, vectorizes it using TF-IDF, and classifies news articles as either real or fake.

## Features

- **Fake News Detection**: Classifies news headlines as real or fake based on text analysis.
- **Preprocessing**: Cleans and processes text data by removing punctuation and numbers.
- **TF-IDF Vectorization**: Converts text into numerical representations using TF-IDF.
- **Machine Learning Model**: Uses Logistic Regression to classify fake and real news.
- **Performance Evaluation**: Provides accuracy, precision, recall, and F1-score metrics.
- **Data Visualization**: Generates a confusion matrix and word clouds for fake and real news.

## Requirements

### Prerequisites

- **Python 3.x**
- Install required dependencies with:

```bash
pip install pandas numpy scikit-learn nltk seaborn matplotlib wordcloud
```

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/fake-news-detector.git
cd fake-news-detector
```

2. Run the application:

```bash
python fake_news_detector.py
```

## Code Structure

### **Preprocessing Function**
The `preprocess_text` function:
- Converts text to lowercase.
- Removes punctuation and numbers.
- Tokenizes and cleans text using regular expressions.

### **Dataset Creation**
The `create_sample_dataset` function generates a small synthetic dataset of fake and real news headlines for testing.

### **Training and Evaluation**
The `train_fake_news_detector` function:
- Preprocesses the dataset.
- Splits the data into training and testing sets.
- Vectorizes text using TF-IDF.
- Trains a Logistic Regression model.
- Evaluates performance using accuracy, precision, recall, and F1-score.
- Visualizes results with a confusion matrix and word clouds.

## Example Workflow

1. **Dataset**: A sample dataset of fake and real news is created.
2. **Preprocessing**: The text is cleaned and prepared for analysis.
3. **Vectorization**: The text is transformed into numerical features.
4. **Model Training**: A Logistic Regression model is trained on the data.
5. **Prediction & Evaluation**: The model predicts fake or real news and generates performance metrics.
6. **Visualization**: A confusion matrix and word clouds display insights from the data.

## Example Output

- **Accuracy Score**: Displays the model's accuracy.
- **Classification Report**: Provides precision, recall, and F1-score.
- **Confusion Matrix**: A heatmap visualization of classification results.
- **Word Cloud**: Visual representation of frequently used words in fake and real news.

## Screenshots

(Add confusion matrix and word cloud images here)

## Troubleshooting

- **Low Accuracy**: Ensure sufficient training data and proper preprocessing.
- **Vectorization Issues**: Confirm that TF-IDF transformation is correctly applied.
- **Incorrect Classifications**: Experiment with different ML models or hyperparameter

