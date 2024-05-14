import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

st.title("Classification Models")

datasetNames = st.sidebar.selectbox("Select Dataset", ["Iris", "Breast Cancer", "Wine", "Digits"])

classifier_names = st.sidebar.selectbox("Select Classifier", [
    "KNN", "SVM", "Random Forest", "Logistic Regression", "Decision Tree",
    "Gradient Boosting", "AdaBoost", "Bagging", "GaussianNB", "MultinomialNB",
    "BernoulliNB", "LDA", "QDA", "MLP", "XGBoost"
])

def getDataset(datasetName):
    if datasetName == "Iris":
        from sklearn.datasets import load_iris
        dataset = load_iris()
    elif datasetName == "Breast Cancer":
        from sklearn.datasets import load_breast_cancer
        dataset = load_breast_cancer()
    elif datasetName == "Wine":
        from sklearn.datasets import load_wine
        dataset = load_wine()
    elif datasetName == "Digits":
        from sklearn.datasets import load_digits
        dataset = load_digits()
    x = dataset.data
    y = dataset.target
    return x, y

x, y = getDataset(datasetNames)

st.write("Shape of the dataset:", x.shape)
st.write("Number of classes:", len(np.unique(y)))

def add_parameter(clf_name):
    params = dict()
    if clf_name == "KNN":
        K = st.sidebar.slider("K", 1, 15)
        params["K"] = K
    elif clf_name == "SVM":
        C = st.sidebar.slider("C", 0.01, 10.0)
        params["C"] = C
    elif clf_name == "Random Forest":
        max_depth = st.sidebar.slider("max_depth", 2, 15)
        n_estimators = st.sidebar.slider("n_estimators", 1, 100)
        params["max_depth"] = max_depth
        params["n_estimators"] = n_estimators
    elif clf_name == "Logistic Regression":
        C = st.sidebar.slider("C", 0.01, 10.0)
        params["C"] = C
    elif clf_name == "Decision Tree":
        max_depth = st.sidebar.slider("max_depth", 2, 15)
        params["max_depth"] = max_depth
    elif clf_name == "Gradient Boosting":
        n_estimators = st.sidebar.slider("n_estimators", 1, 100)
        learning_rate = st.sidebar.slider("learning_rate", 0.01, 1.0)
        params["n_estimators"] = n_estimators
        params["learning_rate"] = learning_rate
    elif clf_name == "AdaBoost":
        n_estimators = st.sidebar.slider("n_estimators", 1, 100)
        learning_rate = st.sidebar.slider("learning_rate", 0.01, 1.0)
        params["n_estimators"] = n_estimators
        params["learning_rate"] = learning_rate
    elif clf_name == "Bagging":
        n_estimators = st.sidebar.slider("n_estimators", 1, 100)
        params["n_estimators"] = n_estimators
    elif clf_name == "MLP":
        alpha = st.sidebar.slider("alpha", 0.0001, 1.0)
        params["alpha"] = alpha
    elif clf_name == "XGBoost":
        n_estimators = st.sidebar.slider("n_estimators", 1, 100)
        learning_rate = st.sidebar.slider("learning_rate", 0.01, 1.0)
        max_depth = st.sidebar.slider("max_depth", 2, 15)
        params["n_estimators"] = n_estimators
        params["learning_rate"] = learning_rate
        params["max_depth"] = max_depth
    return params

params = add_parameter(classifier_names)

def get_classifier(clf_name, params):
    if clf_name == "KNN":
        clf = KNeighborsClassifier(n_neighbors=params["K"])
    elif clf_name == "SVM":
        clf = SVC(C=params["C"])
    elif clf_name == "Random Forest":
        clf = RandomForestClassifier(n_estimators=params["n_estimators"], max_depth=params["max_depth"], random_state=1234)
    elif clf_name == "Logistic Regression":
        clf = LogisticRegression(C=params["C"])
    elif clf_name == "Decision Tree":
        clf = DecisionTreeClassifier(max_depth=params["max_depth"])
    elif clf_name == "Gradient Boosting":
        clf = GradientBoostingClassifier(n_estimators=params["n_estimators"], learning_rate=params["learning_rate"])
    elif clf_name == "AdaBoost":
        clf = AdaBoostClassifier(n_estimators=params["n_estimators"], learning_rate=params["learning_rate"])
    elif clf_name == "Bagging":
        clf = BaggingClassifier(n_estimators=params["n_estimators"])
    elif clf_name == "GaussianNB":
        clf = GaussianNB()
    elif clf_name == "MultinomialNB":
        clf = MultinomialNB()
    elif clf_name == "BernoulliNB":
        clf = BernoulliNB()
    elif clf_name == "LDA":
        clf = LinearDiscriminantAnalysis()
    elif clf_name == "QDA":
        clf = QuadraticDiscriminantAnalysis()
    elif clf_name == "MLP":
        clf = MLPClassifier(alpha=params["alpha"])
    elif clf_name == "XGBoost":
        clf = XGBClassifier(n_estimators=params["n_estimators"], learning_rate=params["learning_rate"], max_depth=params["max_depth"])
    return clf

clf = get_classifier(classifier_names, params)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1234)

clf.fit(x_train, y_train)

ypred = clf.predict(x_test)

acc = clf.score(x_test, y_test)

st.write(f"Classifier: {classifier_names}")
st.write(f"Accuracy: {acc}")

pca = PCA(2)
x_projected = pca.fit_transform(x)

x1 = x_projected[:, 0]
x2 = x_projected[:, 1]

fig = plt.figure()
plt.scatter(x1, x2, c=y, alpha=0.8, cmap="viridis")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar()

st.pyplot(fig)
