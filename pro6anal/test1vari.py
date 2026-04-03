# 표준편차와 분산의 중요성
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import koreanize_matplotlib
import seaborn as sns

np.random.seed(42)

# 목표 평균
target_mean = 60
std_dev_small = 10
std_dev_large = 20

#data
class1_raw = np.random.normal(loc=target_mean, scale=std_dev_small, size=100)
class2_raw = np.random.normal(loc=target_mean, scale=std_dev_large, size=100)

# 평균 보정
class1_adj = class1_raw - np.mean(class1_raw) + target_mean
class2_adj = class2_raw - np.mean(class2_raw) + target_mean

# 정수화
# np.clip(배열, 최소값, 최대값) : 배열의 최소값 이하와 최대값 이상 값을 설정값으로 모두 치환. 배열의 길이는 변하지 않음.
class1 = np.clip(np.round(class1_adj), 10, 100).astype(int)
class2 = np.clip(np.round(class2_adj), 10, 100).astype(int)

# 기술 통계
mean1, mean2 = np.mean(class1), np.mean(class2)
std1, std2 = np.std(class1), np.std(class2)
var1, var2 = np.var(class1), np.var(class2)

print("1반 : 성적 편차가 작은 반")
print(f"평균 : {mean1}, 표준편차 : {std1}, 분산 : {var1}")

print("2반 : 성적 편차가 큰 반")
print(f"평균 : {mean2}, 표준편차 : {std2}, 분산 : {var2}")

# 자료 저장
df = pd.DataFrame({
    'class':['1반'] * 100 + ['2반'] * 100,
    'score':np.concatenate([class1, class2])
    })
print(df)
df.to_csv('test1vari.csv', index=False, encoding='utf-8-sig')

# 자료 시각화
x1 = np.random.normal(1,0.05,size=100)
x2 = np.random.normal(2,0.05,size=100)
plt.figure(figsize=(10,6))
plt.scatter(x1,class1,alpha=0.8,label=f"1반(평균={mean1:2f}, 표준편차={std1:2f})")
plt.scatter(x2,class2,alpha=0.8,label=f"2반(평균={mean2:2f}, 표준편차={std2:2f})")
plt.hlines(target_mean, 0.5,2.5,colors='red', linestyles='dashed', label=f"공통평균={target_mean}")
plt.xticks([1,2],['1반', '2반'])
plt.ylabel("시험점수")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,5))
plt.boxplot([class1,class2], label=['1반', '2반'])
plt.grid()
plt.show()

plt.figure(figsize=(10,6))
plt.hist(class1, bins=15, alpha=0.6, label='1반',edgecolor = 'black')
plt.hist(class2, bins=15, alpha=0.6, label='2반',edgecolor = 'blue')
plt.axvline(target_mean, color='red', linestyle='dotted', label=f"공통평균={target_mean}")
plt.xlabel("시험점수")
plt.ylabel('빈도')
plt.legend()
plt.tight_layout()
plt.show()

