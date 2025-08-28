import numpy as np
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator, ClassifierMixin
from collections import Counter

class KMeansClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_clusters=None, random_state=42, n_init=10):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.n_init = n_init
        self.kmeans_ = None
        self.cluster_to_label_ = None
        self.classes_ = None
        
    def fit(self, X, y):
        labels = np.array(y)
        uniq = np.unique(labels)
        k = self.n_clusters if self.n_clusters is not None else len(uniq)
        self.kmeans_ = KMeans(n_clusters=k, random_state=self.random_state, n_init=self.n_init)
        cluster_ids = self.kmeans_.fit_predict(X)
        mapping = {}
        for c in range(k):
            idx = np.where(cluster_ids == c)[0]
            if len(idx) == 0:
                mapping[c] = uniq[0]
            else:
                votes = labels[idx]
                mapping[c] = Counter(votes).most_common(1)[0][0]
        self.cluster_to_label_ = mapping
        self.classes_ = uniq
        return self
    
    def predict(self, X):
        cid = self.kmeans_.predict(X)
        return np.array([self.cluster_to_label_.get(int(c), self.classes_[0]) for c in cid])
