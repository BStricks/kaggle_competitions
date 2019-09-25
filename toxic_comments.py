import numpy as np 
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score


import seaborn as sns
import matplotlib.pyplot as plt
import os

os.chdir('U:\Documents\PyProjects\kaggle\\toxic_comments')     
train = pd.read_csv( "train.csv", header=0)
test = pd.read_csv( "test.csv", header=0)

labels = ['toxic', 'severe_toxic', 'obscene', 'threat','insult', 'identity_hate']
for x in labels:
    print(train[x].sum())

train['none'] = 1-train[labels].max(axis=1)
train['comment_text'].fillna("unknown", inplace=True)
test['comment_text'].fillna("unknown", inplace=True)        
    
vectorizer = TfidfVectorizer(analyzer='word',
                            stop_words='english',
                            ngram_range=(1, 3),
                            max_features=30000,
                            sublinear_tf=True)


X_train = vectorizer.fit_transform(train.comment_text)
X_test = vectorizer.transform(test.comment_text)

logreg = LogisticRegression(C=0.5, 
                            solver='lbfgs', 
                            multi_class='multinomial') 

for label in labels:
    y = train[label]
    # train the model using X_dtm & y
    logreg.fit(X_train, y)
    # compute the training accuracy
    y_pred_X = logreg.predict(X_train)
    print('Training accuracy is {} for {}'.format(accuracy_score(y, y_pred_X),label))


