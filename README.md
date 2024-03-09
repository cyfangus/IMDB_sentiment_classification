# IMDB_sentiment_classification

## Introduction

This repo contains the Python files to run classification on a commonly used IMDB movie reviews dataset.

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

#### Since the dataset is large, it might require more complex algorithms like SVM and Random Forest classification. Let's see how these models perform.

### Support Vector Machines (SVM) (Linear SVC)
- [IMDB_classification_SVM](IMDB_classification_SVM.py)

```
python IMDB_classification_SVM.py
```

By running this model,
1. It reads and preprocesses the data as IMDB_classification_NB.py does.
2. In model training, grid search was performed with cross-validation. In grid search, 2 ranges of n-grams to consider when tokenizing the text is set to (1, 1) and (1, 2), indicating that (1) only unigrams (single words) and (2) includes both unigrams and bigrams (sequences of two consecutive words) are considered. And there were 3 regularization parameters: 0.1, 1, 10. Therefore, the grid search was performed with (2*3) 6 candidates, while 5 folds were fitted in cross-validation.
3. To evaluate the model performance, pairs of best hyperparameters were reported, alongside the corresponding evaluation matrices.

#### Best Hyperparameters: {'classifier__C': 10, 'vectorizer__ngram_range': (1, 2)} 
| Evaluation matrices | scores |
| --- | --- |
| Accuracy| 0.8916| 
| Precision | 0.8933 |
| Recall | 0.8895 |
| F1 Score | 0.8914 |

![image](https://github.com/cyfangus/IMDB_sentiment_classification/assets/123187295/15f6b287-2163-4e00-a8fe-285760f80546)
![image](https://github.com/cyfangus/IMDB_sentiment_classification/assets/123187295/8b17d867-8097-43f2-aca6-42c701ca2629)

### Random Forest Classification
- [IMDB_classification_RF](IMDB_classification_RF.py)
```
python IMDB_classification_RF.py
```

By running this model,
1. It reads and preprocesses the data as the previous models do.
2. Similarly, grid search and cross-validation were carried out for hyperparameter tuning. In grid search, 4 hyperparameters were included: (1) the number of trees in the forest: [50, 100, and 200], (2) the maximum depth of the trees: [None, 10, 20], (3) the minimum number of samples required to split an internal node: [2, 5, 10], and (4) the minimum number of samples needed to be at a leaf node: [1, 2, 4]. With 3-fold cross-validation, the best Hyperparameters are reported as {'classifier__max_depth': None, 'classifier__min_samples_leaf': 2, 'classifier__min_samples_split': 10, 'classifier__n_estimators': 200}.
3. After that, the ROC curve is plotted and the optimal threshold is calculated based on the maximum difference between the TPR and FPR.

With the threshold = 0.4904, the performance matrices:
| Evaluation matrices | scores |
| --- | --- |
| Accuracy| 0.8589| 
| Precision | 0.8503 |
| Recall | 0.8713 |
| F1 Score | 0.8606 |

![image](https://github.com/cyfangus/IMDB_sentiment_classification/assets/123187295/6065a7fc-e535-4e24-80a1-1ad4a3dec700=250x250)
![image](https://github.com/cyfangus/IMDB_sentiment_classification/assets/123187295/b27a0109-8de7-4cb7-b8d3-8583c589ff88)


## Conclusion
- [IMDB_classification_models_comparison](IMDB_classification_models_comparison.py)
```
python IMDB_classification_models_comparison.py
```
Finally, to compare the performance of these models. A plot with all three models' ROC curves are plotted. It shows that the AUC of the linear SVC is the largest, indicating its capability of distinguishing between the positive and negative reviews. 
![image](https://github.com/cyfangus/IMDB_sentiment_classification/assets/123187295/c7d604ca-fbee-4f2a-be03-db0c0e90afff)



## License
This project is licensed under the [MIT License](LICENSE).
