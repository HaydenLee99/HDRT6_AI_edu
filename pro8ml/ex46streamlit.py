# 작업 결과를 간단하게 웹으로 출력하기
# Python Streamlit 라이브러리 사용
# pip install streamlit

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

texts = [
    '광고성 메일을 확인하세요',
    '회의 일정 변경 공지',
    '무료 쿠폰을 지금 사용하세요',
    '중요한 계약 내용을 확인해주세요',
    '지금 할인 중입니다',
    '오늘 업무 일정 다시 확인해 주세요',
    '지금 바로 확인하세요',
    '사내 공지입니다'
]

labels = ['spam','ham','spam','ham','spam','ham','spam','ham']

vect = CountVectorizer()
x = vect.fit_transform(texts)

model = MultinomialNB()
model.fit(x, labels)

# ---Streamil UI ------------------
import streamlit as st

st.title("이메일 분류기(나이브베이즈)")

user_input = st.text_input("이메일 내용을 입력하세요")

if user_input:
    x_new = vect.transform([user_input])
    pred = model.predict(x_new)[0]
    prob = model.predict_proba(x_new)[0]
    spam_prob = prob[model.classes_.tolist().index('spam')]
    ham_prob = prob[model.classes_.tolist().index('ham')]

    st.write(f'예측 결과 : {pred}')
    st.progress(spam_prob if pred == 'spam' else ham_prob)
    st.write(f'확률 결과 → spam:{spam_prob:.2%}, ham:{ham_prob:.2%}')