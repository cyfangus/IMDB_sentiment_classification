# IMDB_sentiment_classification

## Introduction

This repo contains the python files to run classification on a commonly used IMDB movie reviews dataset.

## Contents

This project contains two machine learning models, training from a dataset containing 25k pos/neg movie reviews and testing on another 25k pos/neg reviews. Since the dataset is too big to be stored in this repository, please download from https://ai.stanford.edu/~amaas/data/sentiment/ to run the following model. Make sure to save the files in the same directory where the codes are stored. Otherwise, change the directory in the code so that it reads from the correct path.

### Multinomial Naive Bayes model

- [IMDB_classification_NB](IMDB_classification_NB.py)

```
python IMDB_classification_NB.py
```

By running this model
1. It starts with reading the datasets from 4 folders containing "train dataset labeled positive", "train dataset labeled negative", "test dataset labeled positive", and "test dataset labeled negative".
2. Then the text data are preprocessed by converting into lowercase, removing punctuation, removing English stopwords with NLTK's stopwords, and performing lemmatization.
3. After that, it trains the Naive Bayes model and gets predicted probabilities. The model performance is then evaluated by ROC curve.
4. Lastly, it finds the cut point with the maximum difference between the TPR and FPR as the optimal threshold. Corresponding matrices of the model are then reported and the confusion matrix is plotted to visualize its performance at this threshold.

![image](https://github.com/cyfangus/IMDB_sentiment_classification/assets/123187295/cb21d6fa-3cb4-4c3f-b9cc-e56cceea658d)
![image](https://github.com/cyfangus/IMDB_sentiment_classification/assets/123187295/a03f0a89-6919-4e55-a9e5-58d3cdc53996)

| Evaluation matrices (threshold = 0.4407)  | scores |
| --- | --- |
| Accuracy | 0.8352 |
| Precision | 0.8220 |
| Recall | 0.8558 |
| F1 Score | 0.8385 |

### Support Vector Machines (SVM)

```
python enhancedmodel.py
```

Since the dataset is large, it might require more complex algorithms like SVM.

By running this model,
1. It reads and preprocesses the data as IMDB_classification_NB.py does.
2. 


## License
This project is licensed under the [MIT License](LICENSE).
