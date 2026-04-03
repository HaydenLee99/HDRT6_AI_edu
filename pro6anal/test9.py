# Independent samples t-test
# 정규분포를 따르는 서로 독립인 두 집단의 평균 차이 검정
# 두 집단의 분산ㅇ; 같다는 가정 필요. 등분산성
# 귀무 : 두 집단의 평균은 같다
# 대립 : 두 집단의 평균은 다르다

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import koreanize_matplotlib
import seaborn as sns

# 실습 1 : 남녀의 시험 평균이 우연히 같을 확률은 얼마나 되나?
# 95% 신뢰 구간에서 우연히 발생할 확률이 5% 이상이면 귀무가설 채택.
male = [75, 85, 100, 72.5, 86.5]
female = [63.2, 76, 52, 100, 70]

two_sample = stats.ttest_ind(male, female)
tv, pv = two_sample
print(f"t 검정 통계량 : {tv:.3f}")
print(f"p-value : {pv:.3f}")
# statistic=1.233, pvalue=0.253, df=8.0
# 해석 : p-value 0.253 > 0.05 귀무 채택

#               정규성      이상치
# levene         무관       민감
# bartlett       필수       둔감


# 정규성 만족 여부 확인 : 샤피로
# 정규성을 만족하지 못하는 경우 : 만 휘트니 검정

# 등분산성 만족 여부 확인
leven_stat, leven_p = stats.levene(male, female)
print('p-value : ', leven_p)
# 등분산성을 만족하지 못하는 경우 : stats.ttest_ind(male, female, equal_var = False) 설정