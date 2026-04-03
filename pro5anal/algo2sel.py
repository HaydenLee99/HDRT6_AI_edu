# 정렬 알고리즘
# list 안에 자료를 오름차순으로 정렬

# 1. Selection Sort
#   방법1: 이해 위주
def fin_min_idx(a):

    n = len(a)
    min_idx = 0

    for i in range(1, n):

        if a[i] < a[min_idx]:
            min_idx = i

    return min_idx

def select_sort(a):
    
    result = []

    while a:
        min_idx = fin_min_idx(a)
        value = a.pop(min_idx)
        result.append(value)

    return result

d = [2,4,5,1,3]
# print(fin_min_idx(d))
print(select_sort(d))

#   방법2: 공간 복잡도 고려 -> 입력 리스트만 사용
def select_sort2(a):
    n = len(a)
    for i in range(n-1):
        min_idx = i
        for j in range(i+1, n):
            if a[j] < a[min_idx]:
                min_idx = j
        a[i], a[min_idx]=a[min_idx], a[i]

d = [2,4,5,1,3]
select_sort2(d)
print(d)
