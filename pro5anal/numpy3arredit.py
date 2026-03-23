# 배열에 행, 열 추가 ...
import numpy as np

aa = np.eye(3,3)
print(aa)

bb = np.c_[aa, aa[0]]       # 0열과 동일한 열 추가
print(bb)

cc = np.r_[aa, [aa[2]]]     # 2행과 동일한 행 추가
print(cc)

print('-- append, insert, delete --\n')
a=np.array([1,2,3])
print(a)
b=np.append(a,[4,5],axis=0)
print(b)
c=np.insert(a, 0, [6,7])
print(c)
d=np.delete(a,1)
print(d)

aa = np.arange(1,10).reshape(3,3)
print(aa)
print(np.insert(aa, 1, 99))             # axis가 없어 차원이 축소되어 출력
print(np.insert(aa, 1, 99, axis=0))
print(np.insert(aa, 1, 99, axis=1))

print('조건 연산 where(조건, 참, 거짓) 인덱스 출력')
x=np.array([1,2,3])
y=np.array([4,5,6])
conditionData = np.array([True, False, True])
result = np.where(conditionData, x, y)
print(result)


print(a[np.where(x>=2)])        # np.where(x>=2)는 조건 True인 인덱스 출력
# 배열 결합
kbs = np.concatenate([x,y])
print(kbs)
mbc, sbs = np.split(kbs, 2)
print(mbc, sbs)

print()
a=np.arange(1,17).reshape(4,4)
print(a)
# 배열 좌우로 분할
x1, x2 = np.hsplit(a,2)
print(x1)
print(x2)
# 배열 상하로 분할
x1, x2 = np.vsplit(a,2)
print(x1)
print(x2)

# 복원/비복원 샘플링
li = np.array([1,2,3,4,5,6,7])

# 복원 추출
for _ in range(5):
    print(li[np.random.randint(0, len(li))], end=' ')
# 비복원 추출
import random
print(random.sample(li.tolist(),5))

print(np.random.choice(range(1,46), 6, replace=True))     # 복원
print(np.random.choice(range(1,46), 6, replace=False))    # 비복원
