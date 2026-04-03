# 단순선형회귀
# 상관관계가 약한 경우와 강한 경우로 회귀분석모델을 생성 후 비교

import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

iris = sns.load_dataset('iris')
iris.info()
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 150 entries, 0 to 149
# Data columns (total 5 columns):
#  #   Column        Non-Null Count  Dtype  
# ---  ------        --------------  -----  
#  0   sepal_length  150 non-null    float64
#  1   sepal_width   150 non-null    float64
#  2   petal_length  150 non-null    float64
#  3   petal_width   150 non-null    float64
#  4   species       150 non-null    object 
# dtypes: float64(4), object(1)

print(iris.iloc[:,0:4].corr())
#               sepal_length  sepal_width  petal_length  petal_width
# sepal_length      1.000000    -0.117570      0.871754     0.817941
# sepal_width      -0.117570     1.000000     -0.428440    -0.366126
# petal_length      0.871754    -0.428440      1.000000     0.962865
# petal_width       0.817941    -0.366126      0.962865     1.000000

# 연습 1 : 상관관계가 약한 경우
result1 = smf.ols(formula='sepal_length ~ sepal_width', data=iris).fit()
print(result1.summary())
print('R-squared : ', result1.rsquared)         # 0.0138         -> 0에 가까운 값이므로 유의하지 않다.
print('p-values : ', result1.pvalues.iloc[1])   # 0.1518 > 0.05  -> 이 모델은 유의하지 않다.

# 시각화
# plt.scatter(iris.sepal_width, iris.sepal_length)
# plt.plot(iris.sepal_width, result1.predict(), color='r')
# plt.show()

# 연습 2 : 상관관계가 강한 경우
result2 = smf.ols(formula='sepal_length ~ petal_length', data=iris).fit()
print(result2.summary())
print('R-squared : ', result2.rsquared)         # 0.7599            -> 1에 가까운 값이므로 유의하다.
print('p-values : ', result2.pvalues.iloc[1])   # 1.038e-47 < 0.05  -> 이 모델은 유의하다.

# 시각화
# plt.scatter(iris.petal_length, iris.sepal_length) 
# plt.plot(iris.petal_length, result2.predict(), color='r')
# plt.show()

# 의미가 있는 값이니 새로운 값 예측 진행
print('실제값 : ', iris.sepal_length[:5].values)
print('예측값 : ', result2.predict()[:5])
# 실제값 :  [5.1 4.9 4.7 4.6 5. ]
# 예측값 :  [4.8790946  4.8790946  4.83820238 4.91998683 4.8790946 ]
new_data = pd.DataFrame({'petal_length':[1.1, 841, 0.5, 0.7, 6.4, 7.7]})
# 학습 데이터의 범위 밖 값 (841)은 신뢰할 수 없음(외삽, Extrapolation)
# 학습 데이터의 범위 안 값만 신뢰할 수 있음(내삽, Interpolation)
y_pred = result2.predict(new_data)
print('예측 결과 : ',y_pred.values)
# 예측 결과 :  [  4.75641792 348.21023867   4.51106455   4.59284901   6.92370599   7.45530495]

# 연습 3 : 독립변수를 복수로 사용하는 다중선형회귀
column_select = "+".join(iris.columns.difference(['sepal_length','sepal_width','species']))
result3 = smf.ols(formula='sepal_length ~ ' + column_select, data=iris).fit()
# result3 = smf.ols(formula='sepal_length ~ petal_length + petal_width', data=iris).fit()
print(result3.summary())
print('Adj. R-squared : ', result3.rsquared_adj)    # 0.763    -> 1에 가까운 값이므로 유의하다.
print('p-values : ', result3.pvalues)               
# petal_length 9.414e-13  < 0.05    -> 0에 가까운 값이므로 유의하다. 
# petal_width  0.04827    < 0.05    -> 0에 가까운 값이므로 유의하다. (약소)