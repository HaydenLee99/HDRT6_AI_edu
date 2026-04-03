import pandas as pd
data = pd.read_csv("pass_cross.csv", encoding="euc-kr")
print(data.head())
print(data.shape)

print(data[(data['공부함']==1) & (data['합격']==1)].shape[0])
print(data[(data['공부함']==1) & (data['불합격']==1)].shape[0])

ctab = pd.crosstab(index=data['공부안함'], columns=data['불합격'], margins=True)
ctab.columns = ['합격', '불합격', '행합']
ctab.index = ['공부함', '공부안함', '열합']
print(ctab)

# 기대 도수 = (각행의 주변합) * (각열의 주변합) / 총합          , 총합은 전체표본수이다.