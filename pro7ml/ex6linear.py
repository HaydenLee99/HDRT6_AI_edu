# 방법 4 : 단순 선형회귀 분석 (인과관계가 있다는 가정하에 진행)
from scipy import stats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# IQ에 따른 시험점수 예측
score_iq = pd.read_csv("https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/score_iq.csv")
score_iq.info()
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 150 entries, 0 to 149
# Data columns (total 6 columns):
#  #   Column   Non-Null Count  Dtype
# ---  ------   --------------  -----
#  0   sid      150 non-null    int64
#  1   score    150 non-null    int64
#  2   iq       150 non-null    int64
#  3   academy  150 non-null    int64
#  4   game     150 non-null    int64
#  5   tv       150 non-null    int64
# dtypes: int64(6)
x = score_iq.iq
y = score_iq.score

print('상관계수 : ', np.corrcoef(x,y)[0,1])

# plt.scatter(x,y)
# plt.show()

model = stats.linregress(x,y)
print(model)
# slope0.651, intercept=-2.856, rvalue=0.882, pvalue=2.847e-50, stderr=0.0285, intercept_stderr=3.546
# pvalue=2.847e-50 으로 기울기가 0이 아니다.

plt.scatter(x,y)
plt.plot(x, model.slope*x + model.intercept, c='r')
plt.show()
# predict method 지원하지 않음
# print('점수예측 : ',np.polyval([model.slope, model.intercept], np.array(score_iq['iq'])))
newdf=pd.DataFrame({'iq':[55,66,77,88,150]})
print('점수예측 : ',np.polyval([model.slope, model.intercept], newdf))
