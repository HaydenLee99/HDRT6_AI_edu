import pandas as pd
data = pd.read_csv('bike_dataset.csv')
print(data.head(2), data.shape)    # (10886, 12)
#   'datetime', 'season'(사계절:1,2,3,4),
#   'holiday'(공휴일(1)과 평일(0)), 'workingday'(근무일(1)과 비근무일(0)),
#   'weather'(4종류:Clear(1), Mist(2), Snow or Rain(3), Heavy Rain(4)),
#   'temp'(섭씨온도), 'atemp'(체감온도), 'humidity'(습도), 'windspeed'(풍속),
#   'casual'(비회원 대여량), 'registered'(회원 대여량), 'count'(총대여량)
# 참고 : casual + registered 가 count 임.
print(data.isnull().sum().sum())    # 0

feature = data.drop(['datetime','casual','registered','count'], axis=1)
print(feature.columns)
label = data["count"]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(feature, label, test_size = 0.3, random_state = 12)

# 모델 생성
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(random_state=12)
model.fit(X_train,y_train)
print('특성(변수) 중요도 : ', model.feature_importances_)
# 'season', 'holiday', 'workingday', 'weather', 'temp', 'atemp', 'humidity', 'windspeed'
# 0.05221519 0.00767489 0.0249678  0.05479437 0.15582838 0.15645981 0.26651692 0.28154264
# 'temp', 'atemp', 'humidity', 'windspeed'를 주요 feature로 설정

feature_imp = feature.drop(['season', 'holiday', 'workingday', 'weather'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(feature_imp, label, test_size = 0.3, random_state = 12)
model.fit(X_train,y_train)

pred = model.predict(X_test)
print('예측값 / 실제값 :\n', (pred / y_test).head())
# 예측값 / 실제값 :
#  2492    1.087413
# 7709    1.015748
# 8592    0.443396
# 165     7.000000
# 6508    0.072464
from sklearn.metrics import r2_score
print('결정계수 : ', r2_score(y_test, pred))
# 결정계수 :  -0.3199974506390091 나쁜 성능
print('예측결과 : ', pred[:5])
# 예측결과 :  [311 258 141   7  10]