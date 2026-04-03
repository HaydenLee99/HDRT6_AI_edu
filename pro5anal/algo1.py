# 알고리즘 평가

# Big-O Notation : 알고리즘의 효율성(성능)을 나타내는 지표
# → 입력 크기(n)가 커질 때 연산 횟수가 어떻게 증가하는지 표현

# 공간 복잡도 (Space Complexity) (HW 발전으로 메모리 제한성이 줄어 중요성이 상대적으로 떨어짐)
# 메모리 사용량 관점 : 사용하는 추가 메모리 크기

# 시간 복잡도 (Time Complexity)
# CPU 연산 횟수 관점 : 실행 시간이 얼마나 빠르게 증가하는지

# 시간 복잡도 순서 (빠름 → 느림)
# O(1) < O(log n) < O(n) < O(n log n) < O(n^2) < O(2^n)

#   O(1) : 상수 시간
#   → 배열 인덱스 접근

#   O(log n) : 로그 시간
#   → 이진 탐색 (Binary Search)

#   O(n) : 선형 시간
#   → 단일 반복문
#   for i in range(n):

#   O(n log n)
#   → 효율적인 정렬 (병합 정렬, 퀵 정렬 평균)

#   O(n^2)
#   → 이중 반복문
#   for i in range(n):
#       for j in range(n):

#   O(2^n)
#   → 모든 경우의 수 탐색 (부분집합, 백트래킹)


# 1 ~ n 까지 연속한 숫자의 합을 구하는 알고리즘
    # 누적합 : O(n)
def sum_n(n):
    s = 0
    for i in range(1, n+1):
        s += i
    return s

    # 가우스 덧셈 공식 이용 : O(1)
def sum_g(n):
    return n*(n+1)/2

# 최대값 구하기
data = [17, 32, 92, 88, -99, 5.85]
    # 비교 연산 : O(n)
def maxi(data):
    maxi = data[0]
    for i in range(1,len(data)):
        maxi = data[i] if maxi < data[i] else maxi
    return maxi

# 최대 공약수 구하기 : O(n1 n2중 작은 값)
    # 반복문
def gcdFunc(n1:int, n2:int):
    i = min(n1, n2)
    while True:
        if n1 % i == 0 and n2 % i == 0:
            return i
        i -= 1

    # 유클리드 호제법 알고리즘
    # 두 자연수 n1, n2(n1 > n2)에서
    # 1. r = n1 % n2 이라 할 때,
    # 2. 최대공약수(GCD)는 GCD(n1, n2) = GCD(n2, r) 이 된다.
    # 3. r = 0일 때, n2가 최대공약수.
def gcdFunc_u(n1, n2):
    if n2 == 0:
        return n1
    return gcdFunc_u(n2, n1 % n2)