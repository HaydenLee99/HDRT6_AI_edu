# [Randomforest 문제1] 
# kaggle.com이 제공하는 'Red Wine quality' 분류 ( 0 - 10)
# dataset은 winequality-red.csv 
# Input variables (based on physicochemical tests):
#  1 - fixed acidity
#  2 - volatile acidity
#  3 - citric acid
#  4 - residual sugar
#  5 - chlorides
#  6 - free sulfur dioxide
#  7 - total sulfur dioxide
#  8 - density
#  9 - pH
#  10 - sulphates
#  11 - alcohol
#  Output variable (based on sensory data):
#  12 - quality (score between 0 and 10)

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

df = pd.read_csv("winequality-red.csv")
print(df.head(2))
df.info()
# print(df.isnull().any().unique())

# 독립변수: quality를 제외한 나머지 모든 컬럼
x = df.drop('quality', axis=1)
# 종속변수: quality 컬럼
y = df['quality']

train_x, test_x, train_y, test_y= train_test_split(x, y, test_size=0.2, random_state=42)
print(train_x.shape, test_x.shape, train_y.shape, test_y.shape) 
# (1276, 11) (320, 11) (1276,) (320,)

# 모델 생성
model = RandomForestClassifier(criterion='gini', n_estimators=500, random_state=1, class_weight='balanced')

# 모델 학습
model.fit(train_x, train_y)

# 예측
pred = model.predict(test_x)
print('예측값 : ', pred[:5])    # [5 6 5 6 6]
print('실제값 : ', np.array(test_y[:5]))    # [5 4 5 5 6]
print('맞춘 갯수 : ', sum(test_y == pred))  #  217
print('전체 대비 맞춘 비율 : ', sum(test_y == pred) / len(test_y))  #  0.678125
print("분류 정확도 : ", accuracy_score(test_y, pred))   #  0.678125
print('classification_report : \n', classification_report(test_y, pred))

# 교차 검증 (KFold)
cross_vali = cross_val_score(model, x, y, cv=5)
print(cross_vali)
print('교차 검증 평균 정확도 : ', np.round(np.mean(cross_vali), 5))

print('중요 변수 확인하기 -----')
print('특성(변수) 중요도 : ', model.feature_importances_)
# .feature_importances_ : 각 특성이 예측에 기여한 정도(중요도)를 수치로 표현
# 값의 합은 1.0, 수치가 클수록 해당 변수가 불순도 감소에 더 많이 기여함.

# 시각화
import matplotlib.pyplot as plt
n_features = x.shape[1]
plt.barh(range(n_features), model.feature_importances_, align='center')
plt.xlabel('Feature importance Score')
plt.ylabel('Features')
plt.yticks(np.arange(n_features), x.columns)
plt.ylim(-1, n_features)
plt.show()
plt.close()

# 새로운 와인 데이터가 들어왔을 때 품질 예측
new_wine = pd.DataFrame([[
    7.5, 0.5, 0.3, 2.0, 0.08,
    15.0, 50.0, 0.997, 3.3, 0.6, 10.5
]], columns=x.columns)
new_pred = model.predict(new_wine)
print(f"새로운 와인 샘플의 예측 품질 등급: {new_pred[0]}")