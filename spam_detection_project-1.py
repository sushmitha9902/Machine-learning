
# Spam Detection using TF-IDF + ML + Hybrid Models

import os
import re
import string
import nltk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc
)

from imblearn.over_sampling import SMOTE
import kagglehub

# -------------------------------
# Install/Download requirements
# -------------------------------
nltk.download('stopwords')

# -------------------------------
# Load Dataset
# -------------------------------
path = kagglehub.dataset_download("ozlerhakan/spam-or-not-spam-dataset")

csv_file = os.path.join(path, "spam.csv")
df = pd.read_csv(csv_file, encoding="cp1252")

df = df[['v1', 'v2']]
df.rename(columns={'v1': 'Category', 'v2': 'Message'}, inplace=True)

df['Category'] = df['Category'].map({
    'ham': 0,
    'spam': 1
})

# -------------------------------
# Data Cleaning
# -------------------------------
df_clean = df.copy()
df_clean = df_clean.dropna(subset=['Category'])
df_clean['Category'] = df_clean['Category'].astype(int)
df_clean = df_clean.drop_duplicates()

def basic_clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text.strip()

df_clean['Message'] = df_clean['Message'].apply(basic_clean_text)

stop_words = set(stopwords.words('english'))

def remove_stopwords(text):
    return " ".join([w for w in text.split() if w not in stop_words])

df_clean['Message'] = df_clean['Message'].apply(remove_stopwords)

stemmer = PorterStemmer()

def stem_text(text):
    return " ".join([stemmer.stem(w) for w in text.split()])

df_clean['Message'] = df_clean['Message'].apply(stem_text)

# -------------------------------
# TF-IDF
# -------------------------------
X = df_clean['Message']
y = df_clean['Category']

tfidf = TfidfVectorizer(
    lowercase=True,
    stop_words='english',
    max_features=20000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)

X = tfidf.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# -------------------------------
# SMOTE
# -------------------------------
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

# -------------------------------
# Models
# -------------------------------
nb = MultinomialNB(alpha=0.1)

lr = LogisticRegression(
    C=5,
    max_iter=5000,
    solver='liblinear'
)

svm = LinearSVC(C=1.5)
svm = CalibratedClassifierCV(svm)

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=30,
    random_state=42
)

hybrid_1 = VotingClassifier(
    estimators=[('nb', nb), ('lr', lr)],
    voting='soft'
)

hybrid_2 = VotingClassifier(
    estimators=[('svm', svm), ('rf', rf)],
    voting='soft'
)

hybrid_all = VotingClassifier(
    estimators=[
        ('nb', nb),
        ('lr', lr),
        ('svm', svm),
        ('rf', rf)
    ],
    voting='soft',
    weights=[1, 2, 3, 2]
)

# -------------------------------
# Train
# -------------------------------
models = [
    nb, lr, svm, rf,
    hybrid_1, hybrid_2, hybrid_all
]

for model in models:
    model.fit(X_train_bal, y_train_bal)

# -------------------------------
# Evaluate
# -------------------------------
results = []

def evaluate_model(name, y_true, y_pred):
    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1 Score": f1_score(y_true, y_pred)
    })

predictions = {
    "Naive Bayes": nb.predict(X_test),
    "Logistic Regression": lr.predict(X_test),
    "SVM": svm.predict(X_test),
    "Random Forest": rf.predict(X_test),
    "Hybrid NB + LR": hybrid_1.predict(X_test),
    "Hybrid SVM + RF": hybrid_2.predict(X_test),
    "Hybrid All 4": hybrid_all.predict(X_test)
}

for name, pred in predictions.items():
    evaluate_model(name, y_test, pred)

results_df = pd.DataFrame(results)

print(results_df)

best_model = results_df.sort_values(
    by="F1 Score",
    ascending=False
).iloc[0]

print("\\nBest Model:\\n", best_model)

best_model_name = best_model["Model"]

if best_model_name == "Naive Bayes":
    final_model = nb
elif best_model_name == "Logistic Regression":
    final_model = lr
elif best_model_name == "SVM":
    final_model = svm
elif best_model_name == "Random Forest":
    final_model = rf
elif best_model_name == "Hybrid NB + LR":
    final_model = hybrid_1
elif best_model_name == "Hybrid SVM + RF":
    final_model = hybrid_2
else:
    final_model = hybrid_all

# -------------------------------
# User Prediction
# -------------------------------
def predict_message(message):
    message = message.lower()
    message = re.sub(r'\\d+', '', message)
    message = message.translate(
        str.maketrans('', '', string.punctuation)
    )

    message = " ".join(
        [w for w in message.split() if w not in stop_words]
    )

    message = " ".join(
        [stemmer.stem(w) for w in message.split()]
    )

    vector = tfidf.transform([message])
    prediction = final_model.predict(vector)[0]

    return "Spam" if prediction == 1 else "Not Spam"

msg = input("Enter a message: ")
print("Prediction:", predict_message(msg))

# -------------------------------
# Accuracy Graph
# -------------------------------
plt.figure(figsize=(10,6))
bars = plt.bar(results_df["Model"], results_df["Accuracy"])

for bar in bars:
    yval = bar.get_height()
    plt.text(
        bar.get_x()+bar.get_width()/2,
        yval,
        round(yval,4),
        ha='center'
    )

plt.xticks(rotation=20)
plt.ylabel("Accuracy")
plt.title("Model Accuracy Comparison")
plt.show()

# -------------------------------
# Confusion Matrix
# -------------------------------
y_pred_all = hybrid_all.predict(X_test)

cm = confusion_matrix(y_test, y_pred_all)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Ham","Spam"]
)

disp.plot()
plt.show()

# -------------------------------
# ROC Curve
# -------------------------------
y_prob = hybrid_all.predict_proba(X_test)[:,1]

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label=f"AUC={roc_auc:.2f}")
plt.plot([0,1],[0,1],'--')
plt.legend()
plt.title("ROC Curve")
plt.show()
