import pandas as pd
import numpy as np
import time

# class MMDP:
#     def __init__(self, input_file):
#         f = open(input_file)
#         lines = f.readlines()
#         lines_number = len(lines)
#         self.weights = np.zeros((lines_number, lines_number))

#         for line in lines:
#             split = line.strip().split("\t")
#             split = [ int(split[0])-1, int(split[1])-1, float(split[2]) ]
#             self.weights[split[0], split[1]] = split[2]
#         # pd.read_csv("./typeI/MDPI1_20.txt", sep="\t", names=["i", "j", "w"])

#     def objective(self, solution):
#         f = 0
#         idx = np.ix_(solution, solution)
#         print(idx)
#         print(self.weights[idx].sum()/len(solution))
#         for i in solution:
#             for j in solution:
#                 if i >= j: continue
#                 f += self.weights[i,j]
#         return f/len(solution)

class MMDP:
    def __init__(self, input_file):
        f = open(input_file)
        lines = f.readlines()

        self.weights = {}

        for line in lines:
            split = line.strip().split("\t")
            split = [ int(split[0]), int(split[1]), float(split[2]) ]
            self.weights[f"{split[0]}->{split[1]}"] = split[2]

        self.n = 20

    def objective(self, solution):
        f = 0
        for i in solution:
            for j in solution:
                if(i>=j): continue
                f += self.weights[f"{i}->{j}"]
        return f/len(solution)

    def get_neighs(self, solution):
        neighs = []
        for i in range(1,self.n):
            if i in solution: neighs.append(solution - {i})
            else: neighs.append(solution | {i})
        return neighs

mmdp = MMDP("./typeI/MDPI1_20.txt")
print(mmdp.objective({1,9,11,13,15,17,18}))
neighs = mmdp.get_neighs({1,9,11,13,15,17,18})

for n in neighs:
    print(mmdp.objective(set(n)))

