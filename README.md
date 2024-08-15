# Fake-News-Detection-using-Machine-Learning


This project focuses on detecting fake news using machine learning algorithms. The dataset used consists of news articles with labels indicating whether the news is "TRUE" or "Fake." The following steps outline the approach taken:

#### **Requirements:**
- Python 3.x
- Required Python libraries:
  - pandas
  - numpy
  - re
  - nltk
  - sklearn
  - pickle

#### **Data Preprocessing:**
1. **Reading Data:**
   - The dataset (`IFND.csv`) is loaded using pandas with different encodings to handle potential encoding issues.

2. **Handling Missing Values:**
   - Missing values in the 'Date' column are filled with empty strings to prevent issues during model training.

3. **Text Cleaning:**
   - The text data is cleaned using regular expressions to remove special characters, links, and unnecessary spaces.
   - Stopwords (common words like "the," "is," "in," etc.) are removed to focus on meaningful words.
   - Words are stemmed using the `PorterStemmer` to reduce them to their root form.

#### **Feature Engineering:**
- **TF-IDF Vectorization:**
  - The text data is transformed into numerical features using the TF-IDF (Term Frequency-Inverse Document Frequency) technique.

#### **Model Training:**
1. **Decision Tree Classifier:**
   - The Decision Tree model is trained on the processed data.
   - Accuracy: 91.92%
   
2. **Logistic Regression:**
   - Logistic Regression model is trained for comparison.
   - Accuracy: 93.48%

3. **K-Nearest Neighbors (KNN):**
   - KNN model is trained as another alternative.
   - Accuracy: 87.91%

#### **Prediction:**
- Functions `predict_label_dt`, `predict_label_logistic`, and `predict_label_knn` are provided to predict the label of a new statement using the trained models.

#### **Model Persistence:**
- The trained models and the TF-IDF vectorizer are saved as pickle files (`vectorizer.pkl`, `decision_tree_model.pkl`, `logistic_regression_model.pkl`, and `knn_model.pkl`) for future use.

#### **Usage:**
1. Load the trained models and TF-IDF vectorizer.
2. Use the prediction functions to classify new statements as "TRUE" or "Fake."

#### **Accuracy Summary:**
- **Logistic Regression:** 93.48%
- **Decision Tree:** 91.92%
- **KNN:** 87.91%

This project provides a basic framework for detecting fake news and can be extended with more sophisticated models or larger datasets for improved accuracy.
