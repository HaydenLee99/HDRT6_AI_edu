# Santander Customer Transaction Prediction의 train.csv를 이용한 실습
# 목표 : Santander bank의 고객만족여부 분류 처리
# target(label) : 0은 만족, 1은 불만족 의미

import pandas as pd
# pd.set_option('display.max_columns', None)
import numpy as np
import matplotlib.pyplot as plt
import koreanize_matplotlib
from xgboost import XGBClassifier, plot_importance
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score

df = pd.read_csv("SantanderCustomerTransactionPredictionTrain.csv")
df.info()
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 76020 entries, 0 to 76019
# Columns: 371 entries, ID to TARGET
# dtypes: float64(111), int64(260)

# print(df.head(3))

# 전체 데이터에서 결측 개수 확인
print(df.isna().sum().sum())
# 결측 없음 확인

# 전체 데이터의 만족과 불만족 비율 확인
print(df['TARGET'].value_counts())
unsatisfied_cnt = df[df['TARGET'] == 1].TARGET.count()
total_cnt = df.TARGET.count()
print(f'불만족 비율 : {unsatisfied_cnt / total_cnt * 100:.2f}%')   # 3.96%

print(df.describe())
# var3 표준편차가 너무 커서, 이상치가 있다 판단. 이상치는 중앙값으로 처리.
df['var3'].replace(-999999, 2, inplace=True)

# 식별자인 ID는 제거
df.drop('ID', axis=1, inplace=True)

print(df.describe())
# 상태 양호함

# feature와 label 분리
x_features = df.iloc[:, :-1]
y_label = df.iloc[:, -1]
print(x_features.shape)

# train / test split
x_train, x_test, y_train, y_test = train_test_split(x_features, y_label, test_size=0.2, random_state=0)
train_cnt = y_train.count()
test_cnt = y_test.count()
print(f'학습 데이터 레이블 값 분포 비율 : {y_train.value_counts() / train_cnt}')
print(f'검증 데이터 레이블 값 분포 비율 : {y_test.value_counts() / test_cnt}')

# 모델 생성 및 학습
xgb_clf = XGBClassifier(n_estimators=50, random_state=12, eval_metric='auc', scale_pos_weight = 96 / 4, verbosity=0)

xgb_clf.fit(x_train, y_train, eval_set=[(x_test, y_test)], verbose=False)      

xgb_roc_score = roc_auc_score(y_test, xgb_clf.predict_proba(x_test)[:, 1])
print(f'xgb_roc_score : {xgb_roc_score:.5f}')

pred = xgb_clf.predict(x_test)
print('예측값 : ', pred[:5])
print('실제값 : ', y_test[:5].values)

from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, pred))

# 최적 파라미터 구하기
params = {'max_depth':[5,7], 'min_child_weight':[1,3],'colsample_bytree':[0.5,0.75]}
# max_depth : 트리 깊이, min_child_weight : 관측치 가중치합 최소, colsample_bytree:피처비
gridcv = GridSearchCV(xgb_clf, param_grid=params)
gridcv.fit(x_train, y_train, eval_set=[(x_test, y_test)], verbose=False)
print('grid cv 최적 파라미터 : ', gridcv.best_params_)
xgb_roc_score = roc_auc_score(y_test, gridcv.predict_proba(x_test)[:, 1], average='macro')
print('xgb_roc_score : ', xgb_roc_score)
# grid cv 최적 파라미터 :  {'colsample_bytree': 0.5, 'max_depth': 7, 'min_child_weight': 1}
# xgb_roc_score :  0.8164821626911527

# 위의 최적의 파라미터로 모델 생성
xgb_clf2 = XGBClassifier(
    n_estimators=5, random_state=12, eval_metric='auc', scale_pos_weight = 96 / 4,
    max_depth=7, colsample_bytree=0.5, min_child_weight=1, verbosity=0
    )
xgb_clf2.fit(x_train, y_train, eval_set=[(x_test, y_test)], verbose=False)
xgb_roc_score2 = roc_auc_score(y_test, xgb_clf2.predict_proba(x_test)[:, 1])
print('xgb_roc_score2 : ', xgb_roc_score2)
# xgb_roc_score2 :  0.835401325917089

# 중요 feature 시각화
fig, ax = plt.subplots(1,1, figsize=(10,8))
plot_importance(xgb_clf2, ax=ax, max_num_features=20)
plt.show()
plt.close()