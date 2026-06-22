import streamlit as st 
import pandas as pd 
from collections import Counter
import sqlite3
import joblib
from wordcloud import WordCloud
import matplotlib.pyplot as plt 
import matplotlib

st.set_page_config(
    page_title="금융 민원 분석 AI",
    page_icon="💰",
    layout="wide"
)

matplotlib.rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False



conn = sqlite3.connect("complaints.db")
cursor = conn.cursor()


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



def delete_predictions():
    cursor.execute(
        "DELETE FROM ai_predictions"
    )
    conn.commit()








def predict_complaint(text):
    text_vector = vectorizer.transform([text])

    prediction = model.predict(text_vector)[0]
    probabilities = model.predict_proba(text_vector)[0]

    max_prob = max(probabilities)

    if max_prob < 0.4:
        prediction = "기타"

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
# 💰 금융 민원 분석 AI 대시보드
AI Hub 금융/보험 민원 데이터를 활용한 민원 분류·분석 서비스입니다.
""")




with st.expander("프로젝트 소개 보기"):
    st.markdown("""
    - AI Hub 금융/보험 민원 데이터 29,490건 활용
    - TF-IDF + Logistic Regression 기반 분류
    - 예측 기록 SQLite 저장
    - 키워드 분석 및 워드클라우드 제공
    """)




df = pd.read_csv("data/complaints.csv")
model = joblib.load("complaints_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

col1, col2, col3 = st.columns(3)

col1.metric("전체 민원 수", f"{len(df):,}건")
col2.metric("분류 카테고리", f"{df['label'].nunique()}개")
col3.metric("모델 정확도", "65.74%")




st.markdown("<br><br>", unsafe_allow_html=True)
st.divider()
st.markdown("<br><br>", unsafe_allow_html=True)



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




st.markdown("<br><br>", unsafe_allow_html=True)
st.divider()
st.markdown("<br><br>", unsafe_allow_html=True)




st.subheader("민원 유형별 개수")

label_count = df["label"].value_counts()

desired_order = [
    "상품 가입 및 해지",
    "잔고 및 거래내역",
    "사고 및 보상 문의",
    "이체, 출금, 대출서비스"
]

label_count = label_count.reindex(desired_order)

col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.bar(
        label_count.index,
        label_count.values,
        width=0.4
    )

    ax.set_title(
        "민원 유형별 개수",
        fontsize=11
    )

    plt.xticks(rotation=15)

    st.pyplot(fig)

with col2:
    fig, ax = plt.subplots(figsize=(4, 4))

    ax.pie(
        label_count,
        labels=label_count.index,
        autopct="%1.1f%%",
        textprops={'fontsize': 8}
    )

    ax.set_title(
        "민원 유형 비율",
        fontsize=11
    )

    st.pyplot(fig)


st.markdown("<br><br>", unsafe_allow_html=True)




st.markdown("<br><br>", unsafe_allow_html=True)
st.divider()
st.markdown("<br><br>", unsafe_allow_html=True)



st.subheader("키워드 TOP 10")

keyword_df = pd.read_csv("data/top_keywords.csv")

st.dataframe(keyword_df)


st.markdown("<br><br>", unsafe_allow_html=True)
st.divider()
st.markdown("<br><br>", unsafe_allow_html=True)



st.subheader("민원 자동 요약")

summary_input = st.text_area("요약할 민원 내용을 입력하세요")

if st.button("요약하기"):
    summary = summarize_complaint(summary_input)
    st.info(f"요약 결과 : {summary}")




st.markdown("<br><br>", unsafe_allow_html=True)
st.divider()
st.markdown("<br><br>", unsafe_allow_html=True)




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



st.markdown("<br><br>", unsafe_allow_html=True)
st.divider()
st.markdown("<br><br>", unsafe_allow_html=True)



st.subheader("AI 예측 기록")

if st.button("🗑️ 예측 기록 전체 삭제"):
    delete_predictions()
    st.success("예측 기록이 삭제되었습니다.")

if st.button("예측 기록 불러오기"):
    prediction_df = load_predictions()
    st.dataframe(prediction_df)

    if not prediction_df.empty:
        st.header("AI 예측 유형별 개수")
        prediction_count = prediction_df["prediction"].value_counts()
        st.bar_chart(prediction_count)


        csv = prediction_df.to_csv(
            index=False
        ).encode("utf-8-sig")

        st.download_button(
            label="예측 기록 다운로드",
            data=csv,
            file_name="prediction_history.csv",
            mime="text/csv"
        )        




st.markdown("<br><br>", unsafe_allow_html=True)
st.divider()
st.markdown("<br><br>", unsafe_allow_html=True)



st.subheader("민원 유형별 TOP 키워드")

selected_label = st.selectbox(
    "분석할 민원 유형",
    list(df["label"].unique()),
    key="keyword_label_select"
)

label_keyword_df = pd.read_csv("data/label_keywords.csv")

selected_keyword_df = label_keyword_df[
    label_keyword_df["label"] == selected_label
]

st.dataframe(
    selected_keyword_df[["키워드", "횟수"]]
)




st.markdown("<br><br>", unsafe_allow_html=True)
st.divider()
st.markdown("<br><br>", unsafe_allow_html=True)




st.subheader("민원 유형별 워드클라우드")

word_freq = dict(
    zip(
        selected_keyword_df["키워드"],
        selected_keyword_df["횟수"]
    )
)

wordcloud = WordCloud(
    font_path="C:/Windows/Fonts/malgun.ttf",
    width=800,
    height=400,
    background_color="white"
).generate_from_frequencies(word_freq)

fig, ax = plt.subplots()
ax.imshow(wordcloud)
ax.axis("off")

st.pyplot(fig)