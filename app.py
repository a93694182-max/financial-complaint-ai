import streamlit as st 
import pandas as pd 
from collections import Counter
import sqlite3
import joblib
from wordcloud import WordCloud
import matplotlib.pyplot as plt 




conn = sqlite3.connect("complaints.db")
cursor = conn.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS complaints(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    content TEXT,
    category TEXT
)
               
""")

conn.commit()

cursor.execute("""
CREATE TABLE IF NOT EXISTS ai_predictions(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    content TEXT,
    prediction TEXT
)
""")

conn.commit()




def save_prediction(content, prediction):
    cursor.execute(
        """
        INSERT INTO ai_predictions
        (content, prediction)
        VALUES(?,?)
        """,
        (content, prediction)
    )
    conn.commit()

def load_predictions():
    prediction_df = pd.read_sql_query(
        "SELECT * FROM ai_predictions",
        conn
    )
    return prediction_df


def save_complaint(content, category):
    cursor.execute(
        """
        INSERT INTO complaints
        (content, category)
        VALUES (?, ?)
        """,
        (content, category)
    )

    conn.commit()


def classify_complaint(text):

    if "대출" in text or "금리" in text or "상환" in text:
        return "대출"

    elif "카드" in text or "결제" in text or "취소" in text:
        return "카드"

    elif "보험" in text or "보험금" in text:
        return "보험"

    elif "펀드" in text or "투자" in text:
        return "투자"

    elif "계좌" in text or "이체" in text:
        return "은행업무"

    else:
        return "기타"



def predict_complaint(text):
    text_vector = vectorizer.transform([text])
    prediction = model.predict(text_vector)[0]
    probabilities = model.predict_proba(text_vector)[0]
    return prediction, probabilities




def summarize_complaint(text):
    words = text.split()
    counter = Counter(words)
    top_counter = counter.most_common(3)
    summary=[]
    for word,count in top_counter:
        summary.append(word)
    return "주요 키워드 : " + ",".join(summary)


st.markdown("""
### 프로젝트 소개

본 프로젝트는 금융 민원 데이터를 분석하고
AI를 활용하여 민원 유형을 자동 분류하는
대시보드입니다.

주요 기능

- 민원 데이터 조회
- 키워드 분석
- 민원 유형 분류
- AI 기반 자동 분류
- 분류 결과 저장 및 조회
""")



st.title("금융 민원 분석 AI 대시보드")
st.write("안녕하세요. 금융 민원 데이터 분석 프로그램입니다")

df = pd.read_csv("data/complaints.csv")
model = joblib.load("complaints_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")



st.subheader("민원 검색 및 필터")
keyword = st.text_input("검색어를 입력하세요")
category = st.selectbox(
    "민원 유형을 선택하세요",
    ["전체"] + list(df["label"].unique())
)

filtered_df = df.copy()

if keyword:
    filtered_df = filtered_df[
        filtered_df["content"].str.contains(keyword, na=False)
    ]
    
if category != "전체":
    filtered_df = filtered_df[
        filtered_df["label"] == category
    ]

st.dataframe(filtered_df)





st.subheader("민원 유형별 개수")
label_count = df["label"].value_counts()
st.bar_chart(label_count)

st.subheader("키워드 TOP 10")
text = " ".join(df["content"])
words = text.split()
counter = Counter(words)
top10 = counter.most_common(10)
keyword_df = pd.DataFrame(
    top10,
    columns=["키워드", "횟수"]
)

st.dataframe(keyword_df)



st.subheader("민원 자동 분류")

user_input = st.text_area(
    "민원 내용을 입력하세요"
)

if st.button("분류하기"):

    result = classify_complaint(user_input)

    save_complaint(
        user_input,
        result
    )

    st.success(
        f"예상 민원 유형 : {result}"
    )



st.subheader("민원 자동 요약")

summary_input = st.text_area("요약할 민원 내용을 입력하세요")

if st.button("요약하기"):
    summary = summarize_complaint(summary_input)
    st.info(f"요약 결과 : {summary}")


st.subheader("AI 민원 분류")
ai_input = st.text_area("AI가 분류할 민원 내용을 입력하세요")



if st.button("AI 분류하기"):
    result, probabilities = predict_complaint(ai_input)
    save_prediction(
        ai_input,
        result
    )
    st.success(f"AI 예측 민원 유형 : {result}")

    prob_df = pd.DataFrame({
        "유형": model.classes_,
        "확률": probabilities
    })
    st.subheader("예측 확률")
    st.dataframe(prob_df)
    st.bar_chart(prob_df.set_index("유형"))


st.subheader("AI 예측 기록")
if st.button("예측 기록 불러오기"):
    prediction_df = load_predictions()
    st.dataframe(prediction_df)

    if not prediction_df.empty:
        st.header("AI 예측 유형별 개수")
        prediction_count = prediction_df["prediction"].value_counts()
        st.bar_chart(prediction_count)



st.subheader("민원 유형별 TOP 키워드")

selected_label = st.selectbox(
    "분석할 민원 유형",
    list(df["label"].unique())
)

label_text = " ".join(
    df[df["label"]==selected_label]["content"]
)

words = label_text.split()
counter = Counter(words)
top_counter = counter.most_common(10)
keyword_df = pd.DataFrame(
    top_counter ,
    columns=["키워드","횟수"]
)
st.dataframe(keyword_df)



st.subheader("민원 유형별 워드클라우드")

wordcloud = WordCloud(
    font_path = "C:/Windows/Fonts/malgun.ttf",
    width = 800,
    height = 400,
    background_color="white"
).generate(label_text)

fig, ax = plt.subplots()
ax.imshow(wordcloud)
ax.axis("off")

st.pyplot(fig)


