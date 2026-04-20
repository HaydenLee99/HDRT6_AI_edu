# 나이브 베이즈 분류(Naive Bayes Classifier) : 어떤 데이터가 특정 클래스일 확률이 가장 큰 쪽으로 분류
# 각 특징이 얼마나 그 클래스다운지를 곱해서 제일 그럴듯한 쪽 고르는 모델

import pandas as pd
import numpy as np
df = pd.read_csv("https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/weather.csv")
df.info()
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 366 entries, 0 to 365
# Data columns (total 12 columns):
#  #   Column        Non-Null Count  Dtype  
# ---  ------        --------------  -----  
#  0   Date          366 non-null    object 
#  1   MinTemp       366 non-null    float64
#  2   MaxTemp       366 non-null    float64
#  3   Rainfall      366 non-null    float64
#  4   Sunshine      363 non-null    float64
#  5   WindSpeed     366 non-null    int64  
#  6   Humidity      366 non-null    int64  
#  7   Pressure      366 non-null    float64
#  8   Cloud         366 non-null    int64  
#  9   Temp          366 non-null    float64
#  10  RainToday     366 non-null    object 
#  11  RainTomorrow  366 non-null    object 
# dtypes: float64(6), int64(3), object(3)

# 전처리
df = df.drop('Date', axis=1)
df['Sunshine'] = df['Sunshine'].fillna(df['Sunshine'].mean())

# 범주형 처리 : 'Yes':1, 'No':0
df['RainToday'] = df['RainToday'].map({'Yes':1, 'No':0})
df['RainTomorrow'] = df['RainTomorrow'].map({'Yes':1, 'No':0})
print(df.head(2))

# feature, label 분류
x = df.drop('RainTomorrow', axis=1)     # feature
y = df['RainTomorrow']                  # label

# 학습(80%), 검증(20%) 데이터 분리
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

# naive_bayes 모델 생성 및 학습 - 연속형 데이터 이므로 GaussianNB 사용
# 연속 데이터를 확률로 바꾸기 위해, 각 feature를 클래스별 정규분포로 가정하는게 가우시안 NB
# dtype 연속형
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(x_train, y_train)

# 예측 및 평가
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
pred = model.predict(x_test)
print('예측값 : ', pred[:10])
print('실제값 : ', y_test[:10].values)
print('예측 정확도 : ', accuracy_score(y_test, pred))           # 약 88%
print('confusion matrix :\n', confusion_matrix(y_test, pred))
print('classification_report :\n', classification_report(y_test, pred))

# 교차 검증
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, x, y, cv=5)
print(f'교차검증 결과에서 각 fold: {scores}, 평균: {scores.mean():.4f}')    # 약 81%

# feature 중요도 분석
# GaussianNB member theta_ : 각 feature별 평균
mean_0 = model.theta_[0]    # RainTomorrow가 0인 경우 평균 (강수예정 X)
mean_1 = model.theta_[1]    # RainTomorrow가 1인 경우 평균 (강수예정 O)
importance = np.abs(mean_0 - mean_1)    # 각 feature가 강수예정 X와 O에서 얼마나 차이가 나는지 의미
feat_impo = pd.DataFrame(
    {'feature':x.columns, 'importance':importance}
).sort_values(by='importance', ascending=False)
print('feature importance:\n',feat_impo)
# feature importance:
#       feature  importance
# 5   Humidity   15.756059
# 6   Pressure    6.070088
# 3   Sunshine    3.698378
# 0    MinTemp    3.448954
# 7      Cloud    2.623589
# 2   Rainfall    1.151417
# 1    MaxTemp    0.745820
# 8       Temp    0.296384
# 9  RainToday    0.157575
# 4  WindSpeed    0.094734

import matplotlib.pyplot as plt
import koreanize_matplotlib
plt.figure(figsize=(10,8))
plt.bar(feat_impo['feature'], feat_impo['importance'])
plt.xlabel('feature')
plt.ylabel('importance by mean')
plt.title('feature importance')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 새로운 자료로 예측
new_data = pd.DataFrame([{
    'MinTemp':12.3,
    'MaxTemp':27.0,
    'Rainfall':0.0,
    'Sunshine':10.0,
    'WindSpeed':8.0,
    'Humidity':40,
    'Pressure':1015.0,
    'Cloud':3,
    'Temp':20.0,
    'RainToday':0
}])
new_pred = model.predict(new_data)
print(
    '예측 결과: ', '강수예정' if new_pred == 1 else '강수예정 없음', 
    '강수확률: ', np.round(model.predict_proba(new_data)[0][1]*100,2),'%'
    )