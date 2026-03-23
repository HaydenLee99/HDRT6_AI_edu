# 연산
from pandas import Series, DataFrame
import numpy as np

s1 = Series([1,2,3],index=['a','b','c'])
s2 = Series([4,5,6,7],index=['a','b','d','c'])
print(s1); print(s2)

print(s1 + s2)      # 같은 index 끼리 연산, 불일치시 NaN
print(s1.add(s2))

print(s1.sub(s2))
print(s1.mul(s2))
print(s1.div(s2))

df1 = DataFrame(np.arange(9).reshape(3,3), columns=list('kbs'), index=['서울','대전','부산'])
df2 = DataFrame(np.arange(12).reshape(4,3), columns=list('kbs'), index=['서울','대전','부산','제주'])
print(df1); print(df2)

print(df1 + df2)
print(df1.add(df2, fill_value=0))

print('NaN(결측값) 처리','-'*10)
df = DataFrame([[1.4,np.nan],[7,-4.5],[np.nan, np.nan],[-0.5,-1]],columns=['one','two'])
print(df)

print(df.isnull())
print(df.notnull())

print(df.dropna())
print(df.dropna(how='any'))
print(df.dropna(how='all'))

print(df.dropna(subset=['one']))
print(df.dropna(subset=['two']))

print(df.dropna(axis='rows'))
print(df.dropna(axis='columns'))