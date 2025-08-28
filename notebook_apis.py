import numpy as np
from collections import Counter
from scipy import sparse
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.metrics import accuracy_score, classification_report
from kmeans_classifier import KMeansClassifier

def _to_dense(X):
    return X.toarray() if sparse.issparse(X) else X

def train_and_test_kmeans(xtrain, ytrain, xtest, ytest, n_cluster, random_state=42, n_init=10):
    model = KMeansClassifier(n_clusters=n_cluster, random_state=random_state, n_init=n_init)
    model.fit(xtrain, ytrain)
    ypred = model.predict(xtest)
    acc = accuracy_score(ytest, ypred)
    rep = classification_report(ytest, ypred)
    return ypred, acc, rep, model

def train_and_test_knn(xtrain, ytrain, xtest, ytest, n_neighbors=5):
    if sparse.issparse(xtrain):
        try:
            model = KNeighborsClassifier(n_neighbors=n_neighbors)
            model.fit(xtrain, ytrain)
        except Exception:
            model = KNeighborsClassifier(n_neighbors=n_neighbors)
            model.fit(_to_dense(xtrain), ytrain)
            xtest = _to_dense(xtest)
    else:
        model = KNeighborsClassifier(n_neighbors=n_neighbors)
        model.fit(xtrain, ytrain)
    ypred = model.predict(xtest)
    acc = accuracy_score(ytest, ypred)
    rep = classification_report(ytest, ypred)
    return ypred, acc, rep, model

def train_and_test_decision_tree(xtrain, ytrain, xtest, ytest, max_depth=None, random_state=42):
    xtrain = _to_dense(xtrain)
    xtest = _to_dense(xtest)
    model = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
    model.fit(xtrain, ytrain)
    ypred = model.predict(xtest)
    acc = accuracy_score(ytest, ypred)
    rep = classification_report(ytest, ypred)
    return ypred, acc, rep, model

def train_and_test_naive_bayes(xtrain, ytrain, xtest, ytest, variant="auto"):
    if variant == "gaussian":
        model = GaussianNB()
        xtrain = _to_dense(xtrain)
        xtest = _to_dense(xtest)
    else:
        model = MultinomialNB()
    model.fit(xtrain, ytrain)
    ypred = model.predict(xtest)
    acc = accuracy_score(ytest, ypred)
    rep = classification_report(ytest, ypred)
    return ypred, acc, rep, model
