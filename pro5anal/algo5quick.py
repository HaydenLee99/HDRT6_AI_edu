# 정렬 알고리즘
# list 안에 자료를 오름차순으로 정렬

# 4.퀵 정렬 : 하나의 기준 점을 중심으로 작은 값과 큰 값을 나눠서 각각 정렬 후 합쳐준다.
# 재귀함수 사용

#   방법1: 이해 위주
def quick_sort(a):
    n = len(a)
    if n <= 1:
        return a
    
    pivot = a[-1]

    g1 = [x for x in a[:n-1] if x <= pivot]  # min group
    g2 = [x for x in a[:n-1] if x > pivot]   # max group

    result = quick_sort(g1) + [pivot] + quick_sort(g2)

    return result

d = [5, 2, 8, 1, 7, 3, 4, 6]
print(quick_sort(d))

#   방법2: 일반 알고리즘
def quick_sort2_sub(a, start_idx, end_idx):
    if end_idx - start_idx <= 0:
        return
    
    pivot = a[end_idx]
    i = start_idx

    for j in range(start_idx, end_idx):
        if a[j] <= pivot:
            a[i], a[j] = a[j], a[i]
            i += 1

    a[i], a[end_idx] = a[end_idx], a[i]

    quick_sort2_sub(a, start_idx, i-1)
    quick_sort2_sub(a, i+1, end_idx)

def quick_sort2(a):
    quick_sort2_sub(a, 0, len(a)-1)         # quick_sort2_sub(자료, 시작 인덱스, 끝 인덱스)

d = [5, 2, 8, 1, 7, 3, 4, 6]
quick_sort2(d)
print(d)