from flask import Flask, render_template, request
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

# ML Models
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# Accuracy
from sklearn.metrics import accuracy_score

app = Flask(__name__)

# =========================
# LOAD DATASET
# =========================

data = pd.read_csv("spam.csv", encoding='latin-1')

# Keep required columns
data = data[['v1', 'v2']]

# Rename columns
data.columns = ['label', 'message']

# Remove null values
data = data.dropna()

# Convert messages into string
data['message'] = data['message'].astype(str)

# Convert labels into numbers
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# =========================
# INPUT AND OUTPUT
# =========================

x = data['message']
y = data['label']

# =========================
# SPLIT DATASET
# =========================

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)

# =========================
# TEXT VECTORIZATION
# =========================

cv = CountVectorizer()

x_train = cv.fit_transform(x_train)
x_test = cv.transform(x_test)

# =========================
# MACHINE LEARNING MODELS
# =========================

models = {
    "Logistic Regression": LogisticRegression(),
    "Naive Bayes": MultinomialNB(),
    "SVM": SVC(),
    "Random Forest": RandomForestClassifier()
}

# =========================
# TRAIN MODELS
# =========================

accuracy_results = {}

for name, model in models.items():

    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)

    accuracy_results[name] = round(accuracy * 100, 2)

# =========================
# HOME PAGE
# =========================

@app.route('/')
def home():

    return render_template(
        'index.html',
        accuracies=accuracy_results
    )

# =========================
# PREDICTION
# =========================

@app.route('/predict', methods=['POST'])
def predict():

    message = request.form['message']

    selected_model = request.form['model']

    data = [message]

    vect = cv.transform(data)

    # Selected ML model
    model = models[selected_model]

    prediction = model.predict(vect)

    if prediction[0] == 1:
        result = "Spam Message"
    else:
        result = "Not Spam"

    return render_template(
        'index.html',
        prediction=result,
        model_used=selected_model,
        accuracies=accuracy_results
    )

# =========================
# RUN APPLICATION
# =========================

if __name__ == '__main__':
    app.run(debug=True)


#Spam Message:
#1.Congratulations! You have won a free iPhone. Click here to claim now.
#2.URGENT! Your bank account has been selected for a cash reward of ₹50,000.
#3.You are the lucky winner of a free vacation package. Reply YES now.
#4.Free entry in a weekly competition. Send WIN to 56789 now.
#5.Claim your exclusive prize before midnight. Limited offer only.
#6.Your mobile number has won a lottery worth ₹10 lakh.
#7.Get a free recharge today. Click the link immediately.
#8.You have been selected for a special cash bonus. Verify your details now.

#Normal/Not Spam Message:
#1.Hi, are you coming to college tomorrow?
#2.Please send me the assignment notes.
#3.Can we meet at 5 PM today?
#4.Mom asked you to buy vegetables on the way home.
#5.Your class starts at 9 AM tomorrow.