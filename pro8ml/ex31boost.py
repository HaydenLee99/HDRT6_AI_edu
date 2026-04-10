# XGboost : boosting 알고리즘을 구현한 분류/예측 모델
# Boosting은 약한 분류기에 대해 샘플의 일부를 보완해가며 순차적 학습해 강한 분류기를 만듦

# brest cancer dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import koreanize_matplotlib
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
from lightgbm import LGBMClassifier     # XGboost보다 성능 우수 하나 자료가 적으면 과적합 발생

# 데이터 준비
data = load_breast_cancer()

x=pd.DataFrame(data.data, columns=data.feature_names)
y=data.target

print('label 분포 :', {name:(y==i).sum() for i, name in enumerate(data.target_names)})
# 'malignant(악성)':212, 'benign(양성)':357

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=12, stratify=y)

# 모델 1
xgb_clf = xgb.XGBClassifier(
    booster='gbtree',
    max_depth=6,
    n_estimator=200,
    eval_metrics = 'logloss',
    random_state=42
)
xgb_clf.fit(x_train, y_train)

# 모델 2
lgb_clf = LGBMClassifier(
    n_estimator=200,
    random_state=42,
    verbose=-1
)
lgb_clf.fit(x_train, y_train)

# 예측 및 평가
pred_xgb = xgb_clf.predict(x_test)
pred_lgb = lgb_clf.predict(x_test)
print(f'xgb_clf acc : {accuracy_score(y_test, pred_xgb):.5f}')  # xgb_clf acc : 0.96491
print(f'lgb_clf acc : {accuracy_score(y_test, pred_lgb):.5f}')  # lgb_clf acc : 0.99123

# feature 중요도 : gain 기준으로 통일
booster = xgb_clf.get_booster()
xgb_gain = pd.Series(booster.get_score(importance_type='gain'))
lgb_gain = pd.Series(
    lgb_clf.booster_.feature_importance(importance_type='gain'),
    index=x_train.columns
)

# print(xgb_gain)
# print(lgb_gain)
# xgb_gain / xgb_gain.sum() : 각 피처의 기여도를 비율로 나타내기
xgb_gain_pct = 100*xgb_gain/(xgb_gain.sum() if xgb_gain.sum() != 0 else 1)
lgb_gain_pct = 100*lgb_gain/(lgb_gain.sum() if lgb_gain.sum() != 0 else 1)

# 사용되지 않은 feature는 0으로 처리
xgb_gain_pct = xgb_gain_pct.reindex(x_train.columns).fillna(0)
lgb_gain_pct = lgb_gain_pct.reindex(x_train.columns).fillna(0)

comp_df = (pd.DataFrame({
    'XGBoost (gain %)':xgb_gain_pct,
    'LightGBM (gain %)':lgb_gain_pct
})).sort_values('XGBoost (gain %)', ascending=False)
print(comp_df.head(10))     # 중요 feature top-10

# 시각화
topk=10
top = comp_df.head(topk)[::-1]
fig, axes = plt.subplots(1,2,figsize=(8,5))
xmax = float(np.ceil(top.max().max()))      # 두 모델의 최대값

for ax, col in zip(axes,['XGBoost (gain %)', 'LightGBM (gain %)']):
    ax.barh(top.index, top[col])
    ax.set_title(f'{col.split()[0]} Feature importance')
    ax.set_xlabel('importance (%)')
    ax.set_xlim(0, xmax)

plt.tight_layout()
plt.show()
plt.close()