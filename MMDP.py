import pandas as pd
import random
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
        self._nodes = set()
        self._W = []

        for line in lines:
            split = line.strip().split("\t")
            split = [ int(split[0]), int(split[1]), float(split[2]) ]
            self.weights[f"{split[0]}->{split[1]}"] = split[2]
            self._nodes.add(split[0])
            self._nodes.add(split[1])

        self.n = len(self._nodes)
        self._np_weights = np.zeros((self.n+1, self.n+1))

        for line in lines:
            split = line.strip().split("\t")
            split = [ int(split[0]), int(split[1]), float(split[2]) ]
            self._np_weights[split[0], split[1]] = split[2]
        print(self._np_weights)

    def objective(self, solution):
        f = 0
        for i in solution:
            for j in solution:
                if(i>=j): continue
                f += self.weights[f"{i}->{j}"]
        return f/len(solution)

    def np_objective(self, solution):
        s = list(solution)
        idx = np.ix_(s, s)
        f = self._np_weights[idx].sum()
        f /= len(s)
        return f

    def get_neighs(self, solution):
        neighs = []
        len_solution = len(solution)
        for i in range(1,self.n+1):
            if i in solution and len_solution > 2: neighs.append(solution - {i})
            else: neighs.append(solution | {i})
        return neighs

    def get_neighs_fast(self, solution):
        neighs = []
        M = len(solution[1])
        if len(self._W) == 0:
            bool_arr = [ i in solution[1] for i in range(0, self.n+1) ]
            self._W = [ self._np_weights[i, bool_arr].sum() for i in range(1, self.n+1) ]

        for i in range(1,self.n+1):
            if i in solution[1] and M > 2:
                new_solution = solution[1] - {i}
                delta = (solution[0]-self._W[i-1])/(M-1)
            else:
                new_solution = solution[1] | {i}
                delta = (-solution[0]+self._W[i-1])/(M+1)

            neighs.append((delta, new_solution))

        for j,p in enumerate(self._W):
            bool_arr = [ i in new_solution for i in range(0, self.n+1) ]
            if (j+1) in new_solution:
                self._W[j] = p - self._np_weights[j,bool_arr].sum()
            else:
                self._W[j] = p + self._np_weights[j,bool_arr].sum()
        return neighs

    def get_random_solution(self):
        return set(random.sample(self._nodes, random.randint(2, self.n)))

# mmdp = MMDP("./instances/typeI/MDPI1_20.txt")
# print(mmdp.objective({1,9,11,13,15,17,18}))
# neighs = mmdp.get_neighs({1,9,11,13,15,17,18})

# for n in neighs:
#     print(mmdp.objective(set(n)))

