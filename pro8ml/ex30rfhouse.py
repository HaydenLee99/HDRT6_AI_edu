# RandomForest는 분류, 회귀 모두 가능. sklearn 모듈은 대개 그러하다.
# 캘리포니아 하우징 data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import koreanize_matplotlib
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

# 데이터 읽어오기
housing = fetch_california_housing(as_frame=True)
# print(housing.DESCR)
# print(housing.data[:2])
# print(housing.target[:2])
# print(housing.feature_names[:2])
df = housing.frame
df.info()
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 20640 entries, 0 to 20639
# Data columns (total 9 columns):
#  #   Column       Non-Null Count  Dtype  
# ---  ------       --------------  -----  
#  0   MedInc       20640 non-null  float64
#  1   HouseAge     20640 non-null  float64
#  2   AveRooms     20640 non-null  float64
#  3   AveBedrms    20640 non-null  float64
#  4   Population   20640 non-null  float64
#  5   AveOccup     20640 non-null  float64
#  6   Latitude     20640 non-null  float64
#  7   Longitude    20640 non-null  float64
#  8   MedHouseVal  20640 non-null  float64
# dtypes: float64(9)

# feature, label 선언
x = df.drop('MedHouseVal', axis=1)
y = df['MedHouseVal']

# 학습, 검증 데이터 분리
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3, random_state=42)

# 모델 생성
rfmodel = RandomForestRegressor(n_estimators=200, random_state=42)

# 모델 학습
rfmodel.fit(x_train, y_train)

# 모델 예측 및 평가
y_pred = rfmodel.predict(x_test)
print(f"MSE : {mean_squared_error(y_test, y_pred):.4f}")    # MSE : 0.2536
print(f"R2_score : {r2_score(y_test, y_pred):.4f}")         # R2_score : 0.8068

# 변수의 기여도 시각화
importances = rfmodel.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(8,5))
plt.bar(range(x.shape[1]), importances[indices], align='center')
plt.xticks(range(x.shape[1]), x.columns[indices], rotation=45)
plt.xlabel('feature name')
plt.ylabel('feature importances')
plt.tight_layout()
plt.show()
plt.close()

# 중요 변수 순위 정보 저장
ranking = pd.DataFrame({
    'feature':x.columns[indices],
    'importance':importances[indices]
})
# print(ranking)
#       feature  importance
# 0      MedInc    0.525400
# 1    AveOccup    0.138819
# 2   Longitude    0.086695
# 3    Latitude    0.086512
# 4    HouseAge    0.054694
# 5    AveRooms    0.045933
# 6  Population    0.032089
# 7   AveBedrms    0.029859

# hyper parameter tuning - RandomizedSearchCV : 사용자가 지정한 범위, 분포에서 임의로 일부 혼합만 샘플링해 탐색
#                                               연속적 값 범위도 사용 가능. 무작위이기에 최적 조합을 못 찾을 수 있음.
from sklearn.model_selection import RandomizedSearchCV
param_dist={
    'n_estimators':[200,400,800],
    'max_depth':[None,10,20,30],
    'min_samples_leaf':[1,2,4],                        # leaf node에 필요한 최소 샘플 수
    'min_samples_split':[1,2,4],                       # node 분할에 필요한 최소 샘플 수
    'max_features':[None,'sqrt','log2',1.0,0.8,0.6]    # 분할 시 고려할 최대 특성 수
}

search = RandomizedSearchCV(
    RandomForestRegressor(random_state=42),
    param_distributions=param_dist,
    n_iter=20,          # 20개의 랜덤한 파라미터 조합을 진행
    scoring='r2',
    cv=3,
    random_state=42,
    verbose=1
)
search.fit(x_train, y_train)
print('best_params : ', search.best_params_)
best = search.best_estimator_
print('best_score : ', search.best_score_)
print('final R2 : ', r2_score(y_test,best.predict(x_test)))