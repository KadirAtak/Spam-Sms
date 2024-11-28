import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import joblib

nltk.download("stopwords")


data = pd.read_csv("spam.csv", encoding="Windows-1252")
data = data.drop(columns=["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)
data = data.rename(columns={"v1": "Etiket", "v2": "SMS"})
data = data.drop_duplicates()
data["Karakter-Sayisi"] = data["SMS"].apply(len)

data.Etiket = [1 if kod == "spam" else 0 for kod in data.Etiket]

def alphabet(cumle):
    yer = re.compile("[^a-zA-Z]")
    return re.sub(yer, " ", cumle)

stop = stopwords.words("english")
all_sentences = []

for i in range(len(data["SMS"].values)):
    r1 = data["SMS"].values[i]
    clean_sentences = []
    sentences = alphabet(r1)
    sentences = sentences.lower()
    for kelimeler in sentences.split():
        clean_sentences.append(kelimeler)
    all_sentences.append(" ".join(clean_sentences))

data["Yeni Sms"] = all_sentences
data = data.drop(columns=["SMS", "Karakter-Sayisi"], axis=1)

cv = CountVectorizer()
xt = cv.fit_transform(data["Yeni Sms"]).toarray()
y = data["Etiket"]
x = xt
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=58)

model = MultinomialNB()
model.fit(X_train, Y_train)
predict = model.predict(X_test)
acs = accuracy_score(Y_test, predict)
print(f"Accuracy: {acs*100:.2f}%")

joblib.dump((model, cv), 'spam_model.pkl')
