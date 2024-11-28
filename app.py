import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
import joblib

# NLTK stopwords indirilmesi
nltk.download("stopwords")

# Veri setinin yüklenmesi ve ön işleme
data = pd.read_csv("spam.csv", encoding="Windows-1252")
data = data.drop(columns=["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)
data = data.rename(columns={"v1": "Etiket", "v2": "SMS"})
data = data.drop_duplicates()
data["Karakter-Sayisi"] = data["SMS"].apply(len)

data.Etiket = [1 if kod == "spam" else 0 for kod in data.Etiket]

# Metin temizleme fonksiyonu
def alphabet(cumle):
    yer = re.compile("[^a-zA-Z]")
    return re.sub(yer, " ", cumle)

stop = stopwords.words("english")
all_sentences = []

# Metinlerin temizlenmesi
for i in range(len(data["SMS"].values)):
    r1 = data["SMS"].values[i]
    clean_sentences = []
    sentences = alphabet(r1)
    sentences = sentences.lower()
    for kelimeler in sentences.split():
        if kelimeler not in stop:  # Stop words filtreleme
            clean_sentences.append(kelimeler)
    all_sentences.append(" ".join(clean_sentences))

data["Yeni Sms"] = all_sentences
data = data.drop(columns=["SMS", "Karakter-Sayisi"], axis=1)

# Özellik çıkarımı
cv = CountVectorizer()
xt = cv.fit_transform(data["Yeni Sms"]).toarray()
y = data["Etiket"]
x = xt

# Eğitim ve test verilerinin ayrılması
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=58)

# Naive Bayes modeli
nb_model = MultinomialNB()
nb_model.fit(X_train, Y_train)
nb_predict = nb_model.predict(X_test)
nb_acs = accuracy_score(Y_test, nb_predict)
print(f"Naive Bayes Accuracy: {nb_acs*100:.2f}%")

# KNN modeli
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, Y_train)
knn_predict = knn_model.predict(X_test)
knn_acs = accuracy_score(Y_test, knn_predict)
print(f"KNN Accuracy: {knn_acs*100:.2f}%")

# SVM modeli
svc_model = SVC(kernel='linear')
svc_model.fit(X_train, Y_train)
svc_predict = svc_model.predict(X_test)
svc_acs = accuracy_score(Y_test, svc_predict)
print(f"SVM Accuracy: {svc_acs*100:.2f}%")

# YSA modeli
y_train_categorical = to_categorical(Y_train, num_classes=2)
y_test_categorical = to_categorical(Y_test, num_classes=2)

ysa_model = Sequential()
ysa_model.add(Dense(512, input_shape=(X_train.shape[1],), activation='relu'))
ysa_model.add(Dense(256, activation='relu'))
ysa_model.add(Dense(2, activation='softmax'))

ysa_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
ysa_model.fit(X_train, y_train_categorical, epochs=10, batch_size=128, verbose=2)

ysa_loss, ysa_accuracy = ysa_model.evaluate(X_test, y_test_categorical, verbose=0)
print(f"YSA Accuracy: {ysa_accuracy*100:.2f}%")

# AdaBoost modeli
ada_model = AdaBoostClassifier(n_estimators=100)
ada_model.fit(X_train, Y_train)
ada_predict = ada_model.predict(X_test)
ada_acs = accuracy_score(Y_test, ada_predict)
print(f"AdaBoost Accuracy: {ada_acs*100:.2f}%")

# Random Forest modeli
rf_model = RandomForestClassifier(n_estimators=100, random_state=58)
rf_model.fit(X_train, Y_train)
rf_predict = rf_model.predict(X_test)
rf_acs = accuracy_score(Y_test, rf_predict)
print(f"Random Forest Accuracy: {rf_acs*100:.2f}%")

# Modellerin kaydedilmesi
joblib.dump((nb_model, cv), 'spam_nb_model.pkl')
joblib.dump((knn_model, cv), 'spam_knn_model.pkl')
joblib.dump((svc_model, cv), 'spam_svc_model.pkl')
ysa_model.save('spam_ysa_model.h5')
joblib.dump((ada_model, cv), 'spam_ada_model.pkl')
joblib.dump((rf_model, cv), 'spam_rf_model.pkl')
