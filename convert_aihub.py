import json
import pandas as pd
import glob

file_paths = glob.glob(
    r"C:\Users\a9369\Downloads\022.민원(콜센터) 질의-응답 데이터\01.데이터\1.Training\라벨링데이터_220121_add\금융보험\**\*.json",
    recursive=True
)

print("찾은 JSON 파일 수:", len(file_paths))

rows = []

for file_path in file_paths:
    print("읽는 중:", file_path)

    with open(file_path, "r", encoding="utf-8-sig") as f:
        data = json.load(f)

    for item in data:
        content = item.get("고객질문(요청)", "")
        label = item.get("카테고리", "")
        speaker = item.get("화자", "")
        qa = item.get("QA", "")

        if speaker == "고객" and qa == "Q" and content and label:
            rows.append({
                "id": len(rows) + 1,
                "content": content,
                "label": label
            })

df = pd.DataFrame(rows)

df.to_csv("data/complaints.csv", index=False, encoding="utf-8-sig")

print("변환 완료")
print("데이터 개수:", len(df))
print("라벨 종류:")
print(df["label"].value_counts())