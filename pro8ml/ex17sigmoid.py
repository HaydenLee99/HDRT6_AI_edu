# Sigmoid function
# Logistic 회귀에서 wx + b는 logit한 값이다.
# log(p/(1-p)) = wx+b
# 그러므로 z = wx+b --> sigmoid(z) --> p(0~1)

import math
import numpy as np
import matplotlib.pyplot as plt
import koreanize_matplotlib

# sigmoid 함수 수식으로 반환된 값 확인 : 양수 -> 0.5보다 큰 값, 음수 -> 0.5보다 작은 값
def sigmoidFunc(num):
    return 1 / (1+math.exp(-num))

print(sigmoidFunc(3))
print(sigmoidFunc(1))
print(sigmoidFunc(-5))
print(sigmoidFunc(-10))

# logit 변환된 값으로 sigmoid 함수 통과 후 그 결과를 시각화
x = np.linspace(-10,10,50)  # 입력 자료 (연속형)
# print(x)

# 선형 결합(이미 logit값)
w = 1.5
bias = -2
z = w*x + bias
def sigmoid(z):
    return 1/(1+np.exp(-z))
p=sigmoid(z)    # 확률값을 얻음
print(p)

# 일부 값 보기
print("x[:3] : ", np.round(x[:3],3))
print("z[:3] : ", np.round(z[:3],3))
print("p[:3] : ", p[:3])

plt.figure(figsize=(12,8))
plt.plot(x,p,label='sigmoid(z)', c='b')
plt.axhline(0.5, c='r', linestyle=':')
plt.title('z = wx + b --> sigmoid --> 확률값')
plt.xlabel('x: 입력값')
plt.ylabel('y: 확률값')
plt.grid(True)
plt.legend()
plt.show()