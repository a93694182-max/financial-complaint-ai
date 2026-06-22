import pandas as pd
from collections import Counter
from konlpy.tag import Okt

STOP_WORDS = [
    "고객", "상담사", "문의", "경우",
    "관련", "이용", "방법", "확인",
    "안내", "처리", "가능",
    "하나요", "얼마", "무엇", "어디",
    "언제", "어떤", "왜", "때문",
    "정도", "부분", "내용", "진행",
    "신청", "발급", "사용", "해지",
    "등록", "변경", "전화", "연락"
]

okt = Okt()

df = pd.read_csv("data/complaints.csv")

def get_keywords(text, top_n=10):
    nouns = okt.nouns(text)

    filtered_nouns = [
        noun for noun in nouns
        if len(noun) >= 2 and noun not in STOP_WORDS
    ]

    counter = Counter(filtered_nouns)

    return counter.most_common(top_n)


# 전체 TOP 키워드
all_text = " ".join(df["content"])
top_keywords = get_keywords(all_text, 10)

top_df = pd.DataFrame(
    top_keywords,
    columns=["키워드", "횟수"]
)

top_df.to_csv(
    "data/top_keywords.csv",
    index=False,
    encoding="utf-8-sig"
)


# 유형별 TOP 키워드
rows = []

for label in df["label"].unique():
    label_text = " ".join(
        df[df["label"] == label]["content"]
    )

    keywords = get_keywords(label_text, 10)

    for word, count in keywords:
        rows.append({
            "label": label,
            "키워드": word,
            "횟수": count
        })

label_keyword_df = pd.DataFrame(rows)

label_keyword_df.to_csv(
    "data/label_keywords.csv",
    index=False,
    encoding="utf-8-sig"
)

print("키워드 파일 생성 완료")
print("전체 키워드:", len(top_df))
print("유형별 키워드:", len(label_keyword_df))