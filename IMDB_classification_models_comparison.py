import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
                             accuracy_score, 
                             precision_score, 
                             recall_score, 
                             f1_score, 
                             roc_curve, 
                             auc,
                             confusion_matrix,
                             ConfusionMatrixDisplay
                             )
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import warnings
import matplotlib.pyplot as plt

# Download NLTK resources (if not already downloaded)
import nltk
nltk.download('stopwords')
nltk.download('wordnet')

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Step 1: Build the functions to read the txt files and preprocess the texts
def read_txt_files(folder_path, label):
    data = []
    labels = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                data.append(content)
                labels.append(label)

    df = pd.DataFrame({'Text': data, 'Label': labels})
    return df

# Text preprocessing function
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    text = ' '.join(word for word in text.split() if word not in stop_words)
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    text = ' '.join(lemmatizer.lemmatize(word) for word in text.split())
    
    return text

# Step 2: Read positive and negative examples into dataframes
train_positive_folder = 'aclImdb/train/pos'
train_negative_folder = 'aclImdb/train/neg'
test_positive_folder = 'aclImdb/test/pos'
test_negative_folder = 'aclImdb/test/neg'

train_positive_df = read_txt_files(train_positive_folder, label=1)
train_negative_df = read_txt_files(train_negative_folder, label=0)
test_positive_df = read_txt_files(test_positive_folder, label=1)
test_negative_df = read_txt_files(test_negative_folder, label=0)

# Combine positive and negative examples into train and test dataframes
train_df = pd.concat([train_positive_df, train_negative_df], ignore_index=True)
test_df = pd.concat([test_positive_df, test_negative_df], ignore_index=True)

# Step 3: Preprocess the text data
train_df['Text'] = train_df['Text'].apply(preprocess_text)
test_df['Text'] = test_df['Text'].apply(preprocess_text)

# Step 4: Create training and testing sets
X_train, y_train = train_df['Text'], train_df['Label']
X_test, y_test = test_df['Text'], test_df['Label']

# Step 5: Create pipelines for each model
nb_pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer()),
    ('classifier', MultinomialNB())
])

svm_pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer(ngram_range=(1, 2))),
    ('classifier', LinearSVC(C=10))
])

rf_pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer()),
    ('classifier', RandomForestClassifier(n_estimators=200, max_depth=None, min_samples_split=10, min_samples_leaf=2))
])

# Step 6: Train the models and get the numbers for plotting ROC
nb_pipeline.fit(X_train, y_train)
svm_pipeline.fit(X_train, y_train)
rf_pipeline.fit(X_train, y_train)

# Step 7: Get the probabilities for ROC curve
y_log_prob_nb = nb_pipeline.predict_log_proba(X_test)[:, 1]
y_prob_nb = np.exp(y_log_prob_nb)
fpr_nb, tpr_nb, thresholds_nb = roc_curve(y_test, y_prob_nb)
roc_auc_nb = auc(fpr_nb, tpr_nb)

decision_values_svm = svm_pipeline.decision_function(X_test)
y_prob_svm = 1 / (1 + np.exp(-decision_values_svm))
fpr_svm, tpr_svm, thresholds_svm = roc_curve(y_test, y_prob_svm)
roc_auc_svm = auc(fpr_svm, tpr_svm)

y_prob_rf = rf_pipeline.predict_proba(X_test)[:, 1]
fpr_rf, tpr_rf, thresholds_rf = roc_curve(y_test, y_prob_rf)
roc_auc_rf = auc(fpr_rf, tpr_rf)

# Plot ROC curves
plt.figure()

plt.plot(fpr_nb, tpr_nb, color='red', lw=2, label='Multinomial Naive Bayes (area = {:.2f})'.format(roc_auc_nb))
plt.plot(fpr_svm, tpr_svm, color='blue', lw=2, label='Linear SVC (area = {:.2f})'.format(roc_auc_svm))
plt.plot(fpr_rf, tpr_rf, color='green', lw=2, label='Random Forest (area = {:.2f})'.format(roc_auc_rf))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()



