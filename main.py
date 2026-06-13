import re
import string
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv("spam.csv", encoding="latin-1")
df = df[["v1", "v2"]].rename(columns={"v1": "label", "v2": "message"})
df.dropna(inplace=True)
df["label_enc"] = df["label"].map({"ham": 0, "spam": 1})

def preprocess(text):
    text = text.lower()
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["clean_message"] = df["message"].apply(preprocess)

X_train, X_test, y_train, y_test = train_test_split(
    df["clean_message"], df["label_enc"],
    test_size=0.2, random_state=42, stratify=df["label_enc"]
)

tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words="english")
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf  = tfidf.transform(X_test)

model = LinearSVC(C=1.0, max_iter=1000)
model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}\n")
print(classification_report(y_test, y_pred, target_names=["Ham", "Spam"]))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

def predict_sms(text):
    vec = tfidf.transform([preprocess(text)])
    return "SPAM" if model.predict(vec)[0] == 1 else "HAM"

test_messages = [
    "Congratulations! You won a free iPhone. Click now to claim your prize!",
    "Hey are you coming to dinner tonight?",
    "URGENT: Your bank account has been suspended. Call 08001234 immediately.",
    "Can you pick up some milk on your way home?",
]

for msg in test_messages:
    print(f"[{predict_sms(msg)}] {msg}")