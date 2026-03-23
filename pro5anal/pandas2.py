# reindex
from pandas import Series, DataFrame
import numpy as np

data = Series([1,3,2], index=(1,4,2))
print(data)

data2 = data.reindex((1,2,4))
print(data2)

print('\n재색인시 값 채워 넣기')
data3 = data2.reindex([0,1,2,3,4,5])
print(data3)

data3 = data2.reindex([0,1,2,3,4,5], fill_value=777)        # fill_value : 대응 값이 없는 index에 이 값을 체움
print(data3)

# NaN을 이전 값으로 체움 : method = ffill or pad
# 첫번째부터 결측이라면 안 채워짐을 유의!
data3 = data2.reindex([0,1,2,3,4,5], method='ffill')
print(data3)
data3 = data2.reindex([0,1,2,3,4,5], method='pad')
print(data3)

print('\nDataFrame : bool 처리')
df = DataFrame(np.arange(1, 13).reshape(4,3), index=['1월','2월','3월','4월'], columns=['강남', '강북', '서초'])
print(df)
print(df['강남'])
print(df['강남'] > 3)
print(df[df['강남'] > 3])

print(df < 3)
df[df < 3] = 0
print(df)

print("\nslicing 관련 method : loc()_label지원, iloc()_int지원")
print(df.loc['3월',:])
print(df.loc[:'2월'])
print(df.loc[:'2월', ['서초']])

print(df.iloc[2])
print(df.iloc[2,:])

print(df.iloc[:3])
print(df.iloc[:3,2])
print(df.iloc[:3,1:3])