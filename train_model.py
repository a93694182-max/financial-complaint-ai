import pandas as pd 
import joblib 


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

df = pd.read_csv("data/complaints.csv")

X = df["content"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

vectorizer = CountVectorizer()

X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

model = MultinomialNB()

model.fit(
    X_train_vectorized,
    y_train
)

print("학습 완료")

test_text = ["대출 금리가 너무 높고 상환이 어렵습니다"]
test_vector = vectorizer.transform(test_text)
prediction = model.predict(test_vector)
print("예측 결과:", prediction[0])


joblib.dump(model, "complaints_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("모델 저장 완료")

y_pred = model.predict(
    X_test_vectorized
)

accuracy = accuracy_score(
    y_test,
    y_pred
)

print(
    "정확도:",
    round(accuracy * 100, 2),
    "%"
)

cm =confusion_matrix(
    y_test,
    y_pred,
    labels=model.classes_
)

print("\n분류 결과")
print(cm)
print("\n라벨 순서")
print(model.classes_)