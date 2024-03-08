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

# Step 5: Create a pipeline for Naive Bayes model
naive_bayes_pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer()),
    ('classifier', MultinomialNB())
])

# Step 6: Train the Naive Bayes model
naive_bayes_pipeline.fit(X_train, y_train)

# Step 7: Get probabilities
y_probabilities = naive_bayes_pipeline.predict_proba(X_test)[:, 1]

# Step 8: Evaluate the performance of the Naive Bayes model using ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_probabilities)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()

# Step 9: Evaluate the performance with threshold-based classification
# Find the optimal threshold and apply
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
y_pred_threshold = (y_probabilities > optimal_threshold).astype(int)

# Report the corresponding metrices
accuracy_threshold = accuracy_score(y_test, y_pred_threshold)
precision_threshold = precision_score(y_test, y_pred_threshold)
recall_threshold = recall_score(y_test, y_pred_threshold)
f1_threshold = f1_score(y_test, y_pred_threshold)

print(f'Accuracy with threshold={optimal_threshold}: {accuracy_threshold:.4f}')
print(f'Precision with threshold={optimal_threshold}: {precision_threshold:.4f}')
print(f'Recall with threshold={optimal_threshold}: {recall_threshold:.4f}')
print(f'F1 Score with threshold={optimal_threshold}: {f1_threshold:.4f}')
    
labels = [0,1]
cm = confusion_matrix(y_test, y_pred_threshold, labels=labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative', 'Positive'])
disp.plot(values_format='')




