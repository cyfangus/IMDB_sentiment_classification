import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
import sklearn.metrics
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
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

# Step 5: Create a pipeline for SVM model
svm_pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer()),
    ('classifier', LinearSVC())
])

# Step 6: Define hyperparameters for grid search
param_grid = {
    'vectorizer__ngram_range': [(1, 1), (1, 2)],
    'classifier__C': [0.1, 1, 10]
}

# Step 7: Perform grid search with cross-validation
grid_search = GridSearchCV(svm_pipeline, param_grid, cv=StratifiedKFold(n_splits=5), scoring='f1', verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Print the best hyperparameters
print("Best Hyperparameters:", grid_search.best_params_)

# Step 8: Train the best model from grid search
best_model = grid_search.best_estimator_
best_model.fit(X_train, y_train)

# Step 9: Make predictions on the test set
y_pred = best_model.predict(X_test)

# Step 10: Evaluate the performance of the best model
accuracy = sklearn.metrics.accuracy_score(y_test, y_pred)
precision = sklearn.metrics.precision_score(y_test, y_pred)
recall = sklearn.metrics.recall_score(y_test, y_pred)
f1 = sklearn.metrics.f1_score(y_test, y_pred)

print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')

# Additional model evaluation metrics
conf_matrix = sklearn.metrics.confusion_matrix(y_test, y_pred)
class_report = sklearn.metrics.classification_report(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=['Negative', 'Positive'])
disp.plot(values_format='')
print("Confusion Matrix:\n", conf_matrix)
print("\nClassification Report:\n", class_report)

# ROC curve and AUC
y_probabilities = best_model.decision_function(X_test)
fpr, tpr, _ = sklearn.metrics.roc_curve(y_test, y_probabilities)
roc_auc = sklearn.metrics.auc(fpr, tpr)

# Precision-Recall curve
precision, recall, _ = sklearn.metrics.precision_recall_curve(y_test, y_probabilities)

# Plot ROC and Precision-Recall curves
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')

plt.subplot(1, 2, 2)
plt.plot(recall, precision, color='blue', lw=2, label='Precision-Recall curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='upper right')

plt.tight_layout()
plt.show()



