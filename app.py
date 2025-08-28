import io
import numpy as np
import pandas as pd
import streamlit as st
import joblib
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.base import BaseEstimator, TransformerMixin
from scipy import sparse
import matplotlib.pyplot as plt
from kmeans_classifier import KMeansClassifier
from pdf_extract import extract_abstract
from embedding_vectorizer import EmbeddingVectorizer

st.set_page_config(page_title="Paper Abstract Topic Classifier", layout="wide")
st.title("Paper Abstract Topic Classifier")

with st.sidebar:
    st.header("Settings")
    vec_choice = st.selectbox("Vectorization", ["BoW", "TF-IDF", "Embedding (E5)"])
    model_choice = st.selectbox("Model", ["KMeans", "KNN", "Decision Tree", "Naive Bayes"])
    random_state = st.number_input("Random state", min_value=0, value=42, step=1)
    max_features = st.number_input("Max features (BoW/TF-IDF)", min_value=100, value=20000, step=100)
    n_neighbors = st.number_input("KNN n_neighbors", min_value=1, value=5, step=1)
    dt_max_depth = st.number_input("Decision Tree max_depth (0=auto)", min_value=0, value=0, step=1)
    km_n_init = st.number_input("KMeans n_init", min_value=1, value=10, step=1)
    embed_model = st.text_input("Embedding model", value="intfloat/multilingual-e5-base")

train_tab, predict_tab = st.tabs(["Train & Evaluate", "Predict from PDF"])

class EmbeddingVectorizerSK(BaseEstimator, TransformerMixin):
    def __init__(self, model_name="intfloat/multilingual-e5-base", normalize=True, device=None, batch_size=32):
        self.model_name = model_name
        self.normalize = normalize
        self.device = device
        self.batch_size = batch_size
        self.inner = None
    def fit(self, X, y=None):
        if self.inner is None:
            self.inner = EmbeddingVectorizer(model_name=self.model_name, normalize=self.normalize, device=self.device, batch_size=self.batch_size)
            try:
                self.inner._ensure_model()
            except Exception:
                pass
        return self
    def transform(self, X):
        return self.inner.transform_numpy(list(X), mode="passage")

def build_vectorizer():
    if vec_choice == "BoW":
        return CountVectorizer(max_features=int(max_features))
    if vec_choice == "TF-IDF":
        return TfidfVectorizer(max_features=int(max_features))
    if vec_choice == "Embedding (E5)":
        return EmbeddingVectorizerSK(model_name=embed_model, normalize=True)

def needs_dense(model_name):
    return model_name in {"Decision Tree"} or (model_name == "Naive Bayes" and vec_choice == "Embedding (E5)")

def build_classifier(num_labels=None):
    if model_choice == "KMeans":
        return KMeansClassifier(n_clusters=num_labels, random_state=int(random_state), n_init=int(km_n_init))
    if model_choice == "KNN":
        return KNeighborsClassifier(n_neighbors=int(n_neighbors))
    if model_choice == "Decision Tree":
        return DecisionTreeClassifier(max_depth=None if int(dt_max_depth) == 0 else int(dt_max_depth), random_state=int(random_state))
    if model_choice == "Naive Bayes":
        return GaussianNB() if vec_choice == "Embedding (E5)" else MultinomialNB()

def to_dense(X):
    return X.toarray() if sparse.issparse(X) else X

with train_tab:
    st.subheader("Upload labeled data and train")
    train_file = st.file_uploader("Upload CSV with columns: abstract, label", type=["csv"])
    c1, c2 = st.columns(2)
    test_size = c1.slider("Test size", 0.05, 0.5, 0.2, 0.05)
    stratify_opt = c2.checkbox("Stratify split", value=True)
    load_model_file = st.file_uploader("Load an existing .joblib model (optional)", type=["joblib"])

    pipeline_obj = None
    loaded_name = None
    if load_model_file is not None:
        pipeline_obj = joblib.load(io.BytesIO(load_model_file.read()))
        loaded_name = load_model_file.name
        st.success(f"Loaded model: {loaded_name}")

    if train_file is not None:
        df = pd.read_csv(train_file)
        cols = df.columns.tolist()
        text_col = st.selectbox("Text column", cols, index=cols.index("abstract") if "abstract" in cols else 0)
        label_col = st.selectbox("Label column", cols, index=cols.index("label") if "label" in cols else min(1, len(cols) - 1))
        X = df[text_col].astype(str).fillna("")
        y = df[label_col].astype(str)
        labels = sorted(y.unique())
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=float(test_size), random_state=int(random_state), stratify=y if stratify_opt else None)
        vec = build_vectorizer()
        clf = build_classifier(num_labels=len(labels))
        steps = [("vectorizer", vec)]
        if needs_dense(model_choice):
            steps.append(("to_dense", FunctionTransformer(to_dense)))
        steps.append(("clf", clf))
        pipeline_obj = Pipeline(steps)
        with st.spinner("Training..."):
            pipeline_obj.fit(Xtr, ytr)
        yhat = pipeline_obj.predict(Xte)
        acc = accuracy_score(yte, yhat)
        st.metric("Accuracy", f"{acc:.4f}")
        st.text(classification_report(yte, yhat))
        cm = confusion_matrix(yte, yhat, labels=labels)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(cm)
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticklabels(labels)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, cm[i, j], ha="center", va="center")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        st.pyplot(fig)

    if pipeline_obj is not None:
        buf = io.BytesIO()
        joblib.dump(pipeline_obj, buf)
        st.download_button("Download model", data=buf.getvalue(), file_name="paper_topic_pipeline.joblib")

with predict_tab:
    st.subheader("Predict topic from PDF")
    if loaded_name:
        st.info(f"Using loaded model: {loaded_name}")
    model_file = st.file_uploader("Or load a model .joblib", type=["joblib"], key="pred_model")
    current_pipeline = None
    if model_file is not None:
        current_pipeline = joblib.load(io.BytesIO(model_file.read()))
    elif 'pipeline_obj' in locals() and pipeline_obj is not None:
        current_pipeline = pipeline_obj
    pdf_file = st.file_uploader("Upload a PDF paper", type=["pdf"])
    manual_text = st.text_area("Or paste an abstract")
    if st.button("Extract & Predict"):
        if current_pipeline is None:
            st.error("Please load or train a model first.")
        else:
            abstract_text = ""
            if pdf_file is not None:
                abstract_text = extract_abstract(pdf_file.read())
            if not abstract_text and manual_text:
                abstract_text = manual_text
            if not abstract_text:
                st.error("No abstract extracted or provided.")
            else:
                st.text_area("Extracted Abstract", abstract_text, height=200)
                yhat = current_pipeline.predict([abstract_text])[0]
                st.success(f"Predicted topic: {yhat}")
                if hasattr(current_pipeline.named_steps["clf"], "predict_proba"):
                    try:
                        proba = current_pipeline.predict_proba([abstract_text])[0]
                        labels = current_pipeline.named_steps["clf"].classes_ if hasattr(current_pipeline.named_steps["clf"], "classes_") else None
                        if labels is not None:
                            proba_df = pd.DataFrame({"label": labels, "probability": proba}).sort_values("probability", ascending=False)
                            st.dataframe(proba_df, use_container_width=True)
                    except Exception:
                        pass
