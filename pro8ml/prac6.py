# [SVM 분류 문제] 심장병 환자 데이터를 사용하여 분류 정확도 분석 연습

# Heart 데이터는 흉부외과 환자 303명을 관찰한 데이터다. 
# 각 환자의 나이, 성별, 검진 정보 컬럼 13개와 마지막 AHD 칼럼에 각 환자들이 심장병이 있는지 여부가 기록되어 있다. 
# dataset에 대해 학습을 위한 train과 test로 구분하고 분류 모델을 만들어, 모델 객체를 호출할 경우 정확한 확률을 확인하시오. 
# 임의의 값을 넣어 분류 결과를 확인하시오.     

# feature 칼럼 : 문자 데이터 칼럼은 제외
# label 칼럼 : AHD(중증 심장질환)

import numpy as np
import pandas as pd
heart = pd.read_csv("https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/Heart.csv")
heart.info()
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 303 entries, 0 to 302
# Data columns (total 15 columns):
#  #   Column      Non-Null Count  Dtype  
# ---  ------      --------------  -----  
#  0   Unnamed: 0  303 non-null    int64  
#  1   Age         303 non-null    int64  
#  2   Sex         303 non-null    int64  
#  3   ChestPain   303 non-null    object 
#  4   RestBP      303 non-null    int64  
#  5   Chol        303 non-null    int64  
#  6   Fbs         303 non-null    int64  
#  7   RestECG     303 non-null    int64  
#  8   MaxHR       303 non-null    int64  
#  9   ExAng       303 non-null    int64  
#  10  Oldpeak     303 non-null    float64
#  11  Slope       303 non-null    int64  
#  12  Ca          299 non-null    float64
#  13  Thal        301 non-null    object 
#  14  AHD         303 non-null    object 
# dtypes: float64(2), int64(10), object(3)

heart.drop(['Unnamed: 0', 'ChestPain', 'Thal'], axis=1, inplace=True)

feature = heart.drop('AHD', axis=1)
feature['Ca'].fillna(feature['Ca'].median(), inplace=True)
# print(feature['Ca'].isna().sum())
# print(feature.columns)
# ['Age', 'Sex', 'RestBP', 'Chol', 'Fbs', 'RestECG', 'MaxHR', 'ExAng', 'Oldpeak', 'Slope', 'Ca']

label = heart['AHD'].map({'Yes':1, 'No':0})

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(feature, label, test_size=0.2, random_state=42)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(x_train)

x_train = sc.transform(x_train)
x_test = sc.transform(x_test)

from sklearn.svm import SVC
model = SVC(C=1.0, kernel='rbf', probability=True).fit(x_train, y_train)

proba = model.predict_proba(x_test)
print('예측값 : ', np.where(proba[:5, 1] > 0.5, 'Yes', 'No'))
print('실제값 : ', np.where(y_test[:5].values > 0.5, 'Yes', 'No'))

from sklearn.metrics import accuracy_score
pred = model.predict(x_test)
sc_score = accuracy_score(y_test, pred)
print('분류 정확도 : ', sc_score)


columns = ['Age', 'Sex', 'RestBP', 'Chol', 'Fbs', 'RestECG', 'MaxHR', 'ExAng', 'Oldpeak', 'Slope', 'Ca']
data = [
    [52, 1, 172, 199, 1, 0, 162, 0, 0.5, 1, 0],
    [47, 0, 138, 250, 0, 1, 150, 1, 1.2, 2, 1],
    [60, 1, 160, 230, 1, 0, 140, 0, 2.3, 1, 2]
]
new_data = pd.DataFrame(data, columns=columns)
new_data = sc.transform(new_data)

new_proba = model.predict_proba(new_data)
print('새로운 값 예측 결과 : ', np.where(new_proba[:, 1] > 0.5, 'Yes', 'No'))