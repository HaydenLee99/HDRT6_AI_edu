# 정렬 알고리즘
# list 안에 자료를 오름차순으로 정렬

# 3.병합 정렬 : 리스트 자료를 반으로 나눔. 요소가 1개씩 남을 때까지 반복 후 다시 붙임
#   방법1: 이해 위주
def merge_sort(a):
    n = len(a)
    if n <= 1:
        return a
    mid = n // 2
    # 함수는 독립적인 공간을 갖는다.
    g1 = merge_sort(a[:mid])
    g2 = merge_sort(a[mid:])
    
    result=[]
    while g1 and g2:
        if g1[0] < g2[0]:
            result.append(g1.pop(0))
        else:
            result.append(g2.pop(0))

    # g1과 g2 중 소진된 것은 skip
    while g1:
        result.append(g1.pop(0))

    while g2:
        result.append(g2.pop(0))

    return result

d = [5, 2, 8, 1, 7, 3, 4, 6]
# print(merge_sort(d))

def merge_sort_debug(a, depth=0):
    indent = "  " * depth
    print(f"{indent}merge_sort called with: {a}")

    n = len(a)
    if n <= 1:
        print(f"{indent}base case reached: {a}")
        return a

    mid = n // 2
    g1 = merge_sort_debug(a[:mid], depth+1)
    g2 = merge_sort_debug(a[mid:], depth+1)

    result = []
    print(f"{indent}start merging g1={g1}, g2={g2}")

    while g1 and g2:
        print(f"{indent}  compare g1[0]={g1[0]} and g2[0]={g2[0]}")
        if g1[0] < g2[0]:
            val = g1.pop(0)
            result.append(val)
            print(f"{indent}  append {val} from g1 → result={result}, g1={g1}, g2={g2}")
        else:
            val = g2.pop(0)
            result.append(val)
            print(f"{indent}  append {val} from g2 → result={result}, g1={g1}, g2={g2}")

    while g1:
        val = g1.pop(0)
        result.append(val)
        print(f"{indent}  append remaining {val} from g1 → result={result}, g1={g1}, g2={g2}")

    while g2:
        val = g2.pop(0)
        result.append(val)
        print(f"{indent}  append remaining {val} from g2 → result={result}, g1={g1}, g2={g2}")

    print(f"{indent}merged result: {result}")
    return result


d = [5, 2, 8, 1, 7, 3, 4, 6]
# merge_sort_debug(d)

#   방법2: 일반 알고리즘
#   재귀 호출이 정렬된 리스트를 반환
#   병합도 새 리스트를 만들어 반환
#   원본 리스트는 그대로이고 정렬된 결과는 새 리스트에 저장
def merge_sort2(a):
    if len(a) <= 1:
        return a
    mid = len(a) // 2
    left = merge_sort2(a[:mid])
    right = merge_sort2(a[mid:])

    result = []
    i = j = 0

    # 병합
    while i < len(left) and j < len(right):
        if left[i] < right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    
    # 남은 요소 추가
    result += left[i:]
    result += right[j:]

    return result

d = [5, 2, 8, 1, 7, 3, 4, 6] 
sorted_d = merge_sort2(d)
print(sorted_d)  
