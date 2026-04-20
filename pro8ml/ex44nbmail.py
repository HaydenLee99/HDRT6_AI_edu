# 스펨 메일 분류기
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
# MultinomialNB : 단어의 빈도를 기반으로 분류, dtype:count

# 학습용 데이터
texts = [
    '무료 쿠폰 잠금 무료 클릭하면 무료 선물 팡팡',
    '한번만 클릭해도 경품 100% 당첨',
    '오늘 회의는 2시야',
    '지금 할인 행사 진행 중',
    '회의 자료는 메일로 보내주세요',
    '지금 바로 쿠폰 확인'
]

labels = ['spam','spam','ham','spam','ham','spam']

# 단어 등장 횟수 기반 벡터
vector = CountVectorizer()
# CountVectorizer : 문서에서 단어 순서정보는 버리고, 단어 빈도수 정보를 뽑아 벡터 표현으로 바꿈
x = vector.fit_transform(texts)
print(vector.get_feature_names_out())
print(vector.vocabulary_)
print(x)    # (문서번호, 단어번호) 등장횟수 꼴

# 모델
from sklearn.metrics import accuracy_score
model = MultinomialNB()
model.fit(x, labels)
pred = model.predict(x)
print('정확도 : ', accuracy_score(labels, pred))

# 새로운 문장 테스트
test_text = ['지금 접속하면 무제한 쿠폰 지급', '명일 간부 회의 시간 및 장소 변경 안내']
x_test = vector.transform(test_text)
print(x_test)

# 예측 + 확률 출력
preds = model.predict(x_test)
probs = model.predict_proba(x_test)
class_names = model.classes_

for text, pred, prob in zip(test_text, preds, probs):
    prob_str = ", ".join(
        [f"{cls}:{p:.4f}" for cls, p in zip(class_names, prob)]
    )
    print(f"'{text}' -> 예측:{pred} / 확률:[{prob_str}]")