# IMDB_sentiment_classification

## Introduction

This repo contains the python files to run classification on a commonly used IMDB movie reviews dataset.

## Contents

This project contains two machine learning models, training from a dataset containing 25k pos/neg movie reviews and testing on another 25k pos/neg reviews. Since the dataset is too big to be stored, please download from https://ai.stanford.edu/~amaas/data/sentiment/ to run the following model. Make sure to save the files in the same directory where the codes store, otherwise change the directory in the code so that it reads from the correct path.

### Naive Bayes model
- [IMDB_classification_NB.py](IMDB_classification_NB.py)

```
python IMDB_classification_NB.py
```

By running this model
1. It starts with reading the datasets from 4 folders containing "train dataset labelled positive", "train dataset labelled negative", "test dataset labelled positive", and "test dataset labelled negative".
2. Then the text data are preprocessed by converting into lowercase, removing puntuation, removing English stopwords with NLTK's stopwords, and performing lemmatization.
3. After that, it trains the Naive Bayes model and get predicted probabilities. The model performace is then evaluated by ROC curve.
4. Lastly, it finds the cut point with the maximum difference between the TPR and FPR as the opimal threshold. Corresponding metrices of the model is then reported and the confusion matrix is ploted to visualise its performance at this thershold.

### Enhanced model

```
python enhancedmodel.py
```

When running this model, an animation should pop up and start running automatically (there is no GUI). 

You should expect to see 20 Sheep (white circles) and 10 Preds (black triangles). The Sheep should move randomly like the agents in the first model, 'eating' the environment as they move and adding to their store, but instead of being sick they stop when they're full (store 200). The Preds make a distance calculation between themselves and each Sheep, moving towards the closest one until they land on the same spot. They will then eat the Sheep, taking the contents of its store into their own, and the Sheep dies. Both classes are plotted based on store size, so as the Sheep eat and the Preds hunt, the points should increase in size. 

To simulate a more realistic representation of reality, the agents in this model no longer wrap round the perimeter, but bounce off the edges. This is a closer representation to what would happen in an actual field with a physical perimeter. 

## License
This project is licensed under the [MIT License](LICENSE).
