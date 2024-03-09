# Data Processing
import os
import numpy as np
import pandas as pd

# Modelling
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# Tree Visualisation
from sklearn.tree import export_graphviz
from IPython.display import Image
import graphviz

# Others
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

# Step 5: Create a pipeline for rf model
rf_pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer()),
    ('classifier', RandomForestClassifier())
])

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'classifier__n_estimators': [50, 100, 200],
    'classifier__max_depth': [None, 10, 20],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(rf_pipeline, param_grid, cv=3, scoring='f1', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Get the best hyperparameters
best_params = grid_search.best_params_
print(f'Best Hyperparameters: {best_params}')

# Train the model with the best hyperparameters
rf_model = grid_search.best_estimator_
y_prob = rf_model.predict_proba(X_test)[:, 1]

# Plot the ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# Step 9: Evaluate the performance with threshold-based classification
# Find the optimal threshold and apply
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
y_pred_threshold = (y_prob > optimal_threshold).astype(int)

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

