# l = [5, 2, 7, 9, 6, 4, 3, 0, 1]


# def first_sort(arr):
#     for i in range(len(arr)):
#         for j in range(len(arr)):
#             if arr[i] < arr[j]:
#                 arr[i], arr[j] = arr[j], arr[i]
#     return arr

# print(first_sort(l))

digits = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

def nested_nest(digits):
    
    """
    A = 2
    B = 1
    C = 7
    D = 8
    """
    for a in digits[1:]:
        for b in digits:
            for c in digits:
                for d in digits[1:]:
                    if 4*(a*1000+b*100+c*10+d*1) == (d*1000+c*100+b*10+a*1):
                        print('A =', a)
                        print('B =', b)
                        print('C =', c)
                        print('D =', d)

nested_nest(digits)