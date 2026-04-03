import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import koreanize_matplotlib
import seaborn as sns

# 독립 표본 t-test 실습
# 두 가지 교육 방법에 따른 평균 시험 점수에 따른 검정 수행 
# 귀무 : 교육 방법에 따른 평균 시험 점수에 차이가 없다.
# 대립 : 교육 방법에 따른 평균 시험 점수에 차이가 있다.

data_raw = pd.read_csv("https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/two_sample.csv")
data = data_raw[['method', 'score']]
# print(data.score.isnull().sum())

data_m1 = data[data.method==1]
# print(data_m1.info())

data_m2 = data[data.method==2]
# print(data_m2.info())

# 교육 방법별 score 추출
score1 = data_m1['score']
# print(score1.isnull().sum())    # 0

score2 = data_m2['score']
# print(score2.isnull().sum())    # 결측치 2개 존재

score2 = score2.fillna(score2.mean())         # 결측치 평균값으로 대체

# 정규성 검증 - 샤피로
print(stats.shapiro(score1))
# statistic=0.965, pvalue=0.367     정규성 만족

print(stats.shapiro(score2))
# statistic=0.962, pvalue=0.671     정규성 만족

sns.histplot(score1, kde=True, color='green')
sns.histplot(score2, kde=True, color='blue')
plt.title('score1&2 정규 분포 확인')
plt.show()

# 등분산성 검증 - levene
print(stats.levene(score1, score2).pvalue)      # p-value 0.456     등분산성 만족

# 정규성과 등분산성을 모두 만족하므로
# 독립 표준 t-test 사용
result = stats.ttest_ind(score1, score2, equal_var=True)
print('result : ', result)
# statistic=-0.196, pvalue=0.845, df=48.0       귀무가설 채택

# 결론 : 유의 수준 0.05 보다 p-value가 매우 크므로 귀무가설 채택함.
# 교육 방법에 따른 평균 시험 성적 차이의 통계적으로 유의한 차이를 발견하지 못함.