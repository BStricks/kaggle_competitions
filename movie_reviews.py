import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set(palette='Blues_d')

os.getcwd()
os.chdir('U:\Documents\PyProjects\kaggle\\movie_review')

train = pd.read_csv('train.tsv',delimiter='\t',encoding='utf-8')
test = pd.read_csv('test.tsv',delimiter='\t',encoding='utf-8')

#check balance of training dataset
proportions = train['Sentiment'].value_counts(sort=False,normalize=True)
ax = sns.barplot(proportions.keys(),proportions.values)
ax.set_title('Proportions of Sentiment labels')
ax.set_ylabel('Proportion')
ax.set_xlabel('Sentiment Labels')

#under-sample
train['Sentiment'].value_counts()
#minority class has 7072, lets randomly sample all classes down
train0 = train.loc[train['Sentiment']==0]
train1 = train.loc[train['Sentiment']==1].sample(n=7072)
train2 = train.loc[train['Sentiment']==2].sample(n=7072)
train3 = train.loc[train['Sentiment']==3].sample(n=7072)
train4 = train.loc[train['Sentiment']==4].sample(n=7072)
train_us = train0.append([train1,train2,train3,train4])
# need to reset index!! otherwise pipeline won't like df
train_us = train_us.reset_index(drop=True)

# data pre-processing stages - can these be wrapped up in count vectorizer pipeline?
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

#count features
count_vectorizer = CountVectorizer(
    input='content',
    encoding='utf-8',
    decode_error='strict',
    strip_accents=None,
    lowercase=True,
    preprocessor=None,
    tokenizer=None,
    stop_words=None,
    token_pattern=r"(?u)\b\w\w+\b",
    ngram_range=(1, 1),
    analyzer='word',
    max_df=1.0,
    min_df=1,
    max_features=None,
    vocabulary=None,
    binary=False,
    dtype=np.int64
)

#tf-idf features
tfidf_vectorizer = TfidfVectorizer(
    input='content',
    encoding='utf-8',
    decode_error='strict',
    strip_accents=None,
    lowercase=True,
    preprocessor=None,
    tokenizer=None,
    stop_words=None,
    token_pattern=r"(?u)\b\w\w+\b",
    ngram_range=(1, 1),
    analyzer='word',
    max_df=1.0,
    min_df=1,
    max_features=None,
    vocabulary=None,
    binary=False,
    dtype=np.int64,
)



# data modelling
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold

svc = LinearSVC(
    C=10,
    class_weight='balanced',
    dual=True,
    fit_intercept=True,
    intercept_scaling=1,
    loss='squared_hinge',
    max_iter=1000,
    multi_class='ovr',
    penalty='l2',
    random_state=0,
    tol=1e-05, 
    verbose=0)

nbc = MultinomialNB()    
rfc = RandomForestClassifier()    


#data is unbalanced, run three versions (original data, under sampled, SMOTE)


pipeline = Pipeline([
    ('count', count_vectorizer),
    ('rf', rfc)])



skf = StratifiedKFold(n_splits=3)

X = train.Phrase
y = train.Sentiment

for traini, testi in skf.split(X, y):
    pipeline.fit(X[traini], y[traini])
    train_score = pipeline.score(X[traini], y[traini])
    test_score = pipeline.score(X[testi], y[testi])
    print("Train = {}, Test = {}".format(train_score, test_score))





"""
first iteration - count 1gram
Train = 0.7349718372133259, Test = 0.5636269270693168
Train = 0.7404075355632449, Test = 0.5427143406382161
Train = 0.7388362392110879, Test = 0.5413318466684609
second iteration - count 2gram
Train = 0.9000749726061631, Test = 0.5720464418899697
Train = 0.9036139946174548, Test = 0.5500192233756248
Train = 0.9018569423886508, Test = 0.5459648583182745
third iteration - count 3gram
Train = 0.9383013898767758, Test = 0.5786205835992465
Train = 0.9400519031141868, Test = 0.5560169165705497
Train = 0.9378039637838566, Test = 0.548963820216079
fourth iteration - tfidf 1gram
Train = 0.72595590072858, Test = 0.5702202914151705
Train = 0.730997693194925, Test = 0.5479815455594003
Train = 0.7297822033409584, Test = 0.5512130416394325
fifth iteration - equal size sampling
Train = 0.8717437420449724, Test = 0.4592875318066158
Train = 0.8708801696712619, Test = 0.474331777683496
Train = 0.8701590668080594, Test = 0.46304624522698346
sixth iteration - count 1 gram, SVM c 0.1
Train = 0.7164209231242431, Test = 0.5872131021490907
Train = 0.720876585928489, Test = 0.5670319108035371
Train = 0.7211991311201246, Test = 0.5657464723749471
sixth iteration - count 1 gram, SVM c 10
Train = 0.7380091889501913, Test = 0.5495367344584984
Train = 0.7423394848135333, Test = 0.5308919646289888
Train = 0.7418350281617039, Test = 0.5346610788573186
sixth iteration - count 1 gram, RF 
"""


X_test=test['Phrase']
pipe_pred = pipeline.predict(X_test)
test['Sentiment'] = pipe_pred
test.to_csv('test.csv', sep=',', index=False)
