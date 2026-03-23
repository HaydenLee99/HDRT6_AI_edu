# 통계 기본 

# 통계량 : 데이터의 특징을 하나의 숫자로 요약한 것.
# 표본 데이터를 추출해 전체(모집단) 데이터 짐작 가능

# 1. 평균 (Mean) : 𝑋̄ (엑스 바)
# : 데이터의 대표값 (중심)

# 2. 편차 (Deviation) : (X - 𝑋̄)
# : 평균으로부터 개별 데이터가 떨어진 정도

# 3. 분산 (Variance) : Var(X) 또는 σ²
# : 편차의 제곱의 평균 (평균으로부터 전체 데이터가 떨어진 정도)

# 4. 표준편차 (Standard Deviation) : σ = √Var(X)
# : 분산에 루트를 씌운 값

grades = [1, 3, -2, 4]      # 별량

def show_grades(grades):
    print("별량에서 ", end=" ")
    for g in grades:
        print(g, end=" ")
    print("\n")
show_grades(grades)

def grades_sum(grades):
    total = 0
    for g in grades:
        total += g
    return total
print("합은 ", grades_sum(grades))

def grades_ave(grades):
    ave = grades_sum(grades) / len(grades)
    return ave
print("평균은 ", grades_ave(grades))

def grades_variance(grades):
    ave = grades_ave(grades)
    dev = 0
    for su in grades:
        dev += (su - ave) ** 2
    var = dev / len(grades)
    return var
print("분산은 ", grades_variance(grades))

def grades_std(grades, n:int=2):         # n은 출력할 소수점 자리수
    return round(grades_variance(grades) ** 0.5, n)
print("표준편차는 ", grades_std(grades))

print("\nnumpy 사용")
import numpy as np
# NumPy: 내부가 C로 구현되어 있어 연산 속도가 빠름
# 파이썬 list보다 연속된 메모리 구조를 사용하여 효율적인 계산 가능
# 벡터화(vectorization)를 통해 반복문 없이 빠른 배열 연산 지원
# slicing, indexing, broadcasting 등 다양한 기능 제공

print("합은 ", np.sum(grades))
print("평균은 ", np.mean(grades))
print("분산은 ", np.var(grades))
print("표준편차는 ", round(np.std(grades), 2))