import test3

# for _ in range(10):
#     print(test3.count)
#     test3.inc()

# x = .05
# print(str(x))

# start, end, step = 0, 46, 5
# xtick_list = []
# for i, x in enumerate(range(start, end, step)):
#     print(i, x, str(x/100))
#     xtick_list.append(str(x/100))
# print(xtick_list)    

# start = 0  # inclusive
# end = 46    # exclusive
# step = 5
# diff = end - 1 - start
# cycles = 50
# max_redos = 30

# # Initialize results matrix - eg: results[1][3] --> Euclidean runtime on graph 4
# results = [[0 for _ in range((end - 1 - start)/5 + 1)] for _ in range(3)]

# print(len(results[1]))

for x, y in enumerate(range(0, 46, 5)):
    print(x,y)