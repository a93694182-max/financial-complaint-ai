import pandas as pd 
import joblib 
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False


df = pd.read_csv("data/complaints.csv")

X = df["content"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

vectorizer = TfidfVectorizer(
    stop_words=["은", "는", "이", "가", "을", "를", "에", "의", "도", "로", "으로", "하고", "하면", "해서"]
)

X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000)
}

best_model = None
best_accuracy = 0
best_model_name = ""

for model_name, model in models.items():
    model.fit(X_train_vectorized, y_train)

    y_pred = model.predict(X_test_vectorized)

    accuracy = accuracy_score(y_test, y_pred)

    print(model_name, "정확도:", round(accuracy * 100, 2), "%")

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model
        best_model_name = model_name
        best_y_pred = y_pred

print("최고 성능 모델:", best_model_name)
print("최고 정확도:", round(best_accuracy * 100, 2), "%")

model = best_model
y_pred = best_y_pred
accuracy = best_accuracy

test_text = ["대출 금리가 너무 높고 상환이 어렵습니다"]
test_vector = vectorizer.transform(test_text)
prediction = model.predict(test_vector)
print("예측 결과:", prediction[0])


joblib.dump(model, "complaints_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("모델 저장 완료")


cm =confusion_matrix(
    y_test,
    y_pred,
    labels=model.classes_
)

print("\n분류 결과")
print(cm)
print("\n라벨 순서")
print(model.classes_)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=model.classes_
)

fig, ax = plt.subplots(figsize=(10, 8))

disp.plot(
    ax=ax,
    xticks_rotation='horizontal'
)

plt.title("Confusion Matrix")
plt.tight_layout()

plt.savefig("confusion_matrix.png")

print("혼동행렬 이미지 저장 완료")
