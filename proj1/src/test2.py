import test3

# for _ in range(10):
#     print(test3.count)
#     test3.inc()

# x = .05
# print(str(x))

start, end, step = 0, 46, 5
xtick_list = []
for i, x in enumerate(range(start, end, step)):
    print(i, x, str(x/100))
    xtick_list.append(str(x/100))
print(xtick_list)    
