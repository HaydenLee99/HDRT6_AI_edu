# logistic regresiion
# 선형결합 로그오즈로 해석하고 이를 시그모이드 함수를 통해 확률로 변환

# 이항분류, 독립변수 연속형 종속변수 범주형
# 뉴럴넷에 뉴런에서 사용

import statsmodels.api as sm

mtcars = sm.datasets.get_rdataset('mtcars').data
mtcars.info()
# <class 'pandas.core.frame.DataFrame'>
# Index: 32 entries, Mazda RX4 to Volvo 142E
# Data columns (total 11 columns):
#  #   Column  Non-Null Count  Dtype  
# ---  ------  --------------  -----  
#  0   mpg     32 non-null     float64
#  1   cyl     32 non-null     int64  
#  2   disp    32 non-null     float64
#  3   hp      32 non-null     int64  
#  4   drat    32 non-null     float64
#  5   wt      32 non-null     float64
#  6   qsec    32 non-null     float64
#  7   vs      32 non-null     int64  
#  8   am      32 non-null     int64  
#  9   gear    32 non-null     int64  
#  10  carb    32 non-null     int64  
# dtypes: float64(5), int64(6)

# 연비와 마력수에 따른 변속기 분류
mtcar = mtcars.loc[:,['mpg','hp','am']]
print(mtcar['am'].unique())     # [1 0] 이항 분류 가능.   1: 수동, 0: 자동

# 모델 작성 방법 1 : logit()
import numpy as np
import statsmodels.formula.api as smf
formula = 'am ~ hp+mpg'     # 연속형 ~ 범주형+...
result = smf.logit(formula=formula, data=mtcar).fit()
print(result.summary())

pred = result.predict(mtcar[:10])
print('예측값:', np.around(pred.values).astype(int))
print('실제값:', mtcar['am'][:10].values)

# 수치에 대한 집계표
conf_table = result.pred_table()
print('confusion matrix , 혼돈행렬\n', conf_table)

# Confusion Matrix (혼동 행렬)
# 0행 0열 (실제 0, 예측 0) → TN : 맞게 예측
# 0행 1열 (실제 0, 예측 1) → FP : 틀리게 예측 (오탐)
# 1행 0열 (실제 1, 예측 0) → FN : 틀리게 예측 (놓침)
# 1행 1열 (실제 1, 예측 1) → TP : 맞게 예측  

# Accuracy (정확도)
# 전체 중에서 맞춘 비율
# (TP + TN) / (TP + TN + FP + FN)

# Precision (정밀도) : 틀리게 양성 예측(FP)을 줄이는 것이 중요
# 맞다고 예측한 것 중 실제로 맞은 비율
# TP / (TP + FP)

# Recall (재현율) : 놓치는 것(FN)을 줄이는 것이 중요
# 실제로 맞는 것 중에서 맞춘 비율
# TP / (TP + FN)

# F1-score : Precision과 Recall의 균형을 평가
# Precision과 Recall의 조화평균
# 2 * (Precision * Recall) / (Precision + Recall)

from sklearn.metrics import accuracy_score
pred2 = result.predict(mtcar)
print('분류정확도 : ', accuracy_score(mtcar['am'], np.around(pred2)))

# 모델 작성 방법 2 : glm() - 일반화된 선형모델
result2 = smf.glm(formula=formula, data=mtcar, family=sm.families.Binomial()).fit()
# Binomial : 이항분포       Gaucian : 정규분포(default)
print(result2.summary())

glm_pred = result2.predict(mtcar[:10])
print('glm 예측값 : ', np.around(glm_pred.values).astype(int))
print('glm 실제값 : ', mtcar['am'][:10].values)

glm_pred2 = result2.predict(mtcar)
print('glm 모델 분류 정확도 : ', accuracy_score(mtcar['am'], np.around(glm_pred2)))

# logit()은 변환 함수, glm()은 logit을 포함한 전체 모델

# 새로운 값으로 분류 작업
import pandas as pd
new_df = pd.DataFrame()
new_df['mpg'] = [10,20,120,200]
new_df['hp'] = [100,110,80,130]
print(new_df)
new_pred = result2.predict(new_df)
print('예측 결과 : ',new_pred.values.astype(int))

