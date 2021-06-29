import pandas as pd
import random
import numpy as np
import time
import torch
GRAPH_SIZE = 35

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
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
    def __init__(self, input_file=None):
        files = [
                f"./instances/typeI/MDPI1_{GRAPH_SIZE}.txt",
                f"./instances/typeII/MDPII1_{GRAPH_SIZE}.txt",
                f"./instances/typeI/MDPI2_{GRAPH_SIZE}.txt",
                f"./instances/typeII/MDPII2_{GRAPH_SIZE}.txt",
                f"./instances/typeI/MDPI3_{GRAPH_SIZE}.txt",
                f"./instances/typeII/MDPII3_{GRAPH_SIZE}.txt",
                f"./instances/typeI/MDPI4_{GRAPH_SIZE}.txt",
                f"./instances/typeII/MDPII4_{GRAPH_SIZE}.txt",
                f"./instances/typeI/MDPI5_{GRAPH_SIZE}.txt",
                f"./instances/typeII/MDPII5_{GRAPH_SIZE}.txt",
                f"./instances/typeI/MDPI6_{GRAPH_SIZE}.txt",
                f"./instances/typeII/MDPII6_{GRAPH_SIZE}.txt",
                f"./instances/typeI/MDPI7_{GRAPH_SIZE}.txt",
                f"./instances/typeII/MDPII7_{GRAPH_SIZE}.txt",
                f"./instances/typeI/MDPI8_{GRAPH_SIZE}.txt",
                f"./instances/typeII/MDPII8_{GRAPH_SIZE}.txt",
                f"./instances/typeI/MDPI9_{GRAPH_SIZE}.txt",
                f"./instances/typeII/MDPII9_{GRAPH_SIZE}.txt",
                f"./instances/typeI/MDPI10_{GRAPH_SIZE}.txt",
                f"./instances/typeII/MDPII10_{GRAPH_SIZE}.txt",
                ]
        if input_file is None:
            input_file = random.sample(files, 1)[0]
        f = open(input_file)
        lines = f.readlines()

        self.weights = {}
        self._nodes = set()

        for line in lines:
            split = line.strip().split("\t")
            split = [ int(split[0]), int(split[1]), float(split[2]) ]
            self.weights[f"{split[0]}->{split[1]}"] = split[2]
            self._nodes.add(split[0]-1)
            self._nodes.add(split[1]-1)

        self.n = len(self._nodes)
        self.W_np = np.zeros((self.n, self.n))

        for line in lines:
            split = line.strip().split("\t")
            split = [ int(split[0])-1, int(split[1])-1, float(split[2]) ]
            self.W_np[split[0], split[1]] = split[2]

        self.W = torch.from_numpy(self.W_np).to(device)

    # def objective(self, solution):
    #     f = 0
    #     for i in solution:
    #         for j in solution:
    #             if(i>=j): continue
    #             f += self.weights[f"{i}->{j}"]
    #     return f/len(solution)

    def objective(self, solution):
        if len(solution) == 0: return 0
        s = torch.tensor(list(solution))-1
        idx = np.ix_(s, s)
        f = self.W[idx].sum()
        f /= len(s)
        return f

    def get_neighs(self, solution):
        neighs = []
        len_solution = len(solution)
        for i in range(1,self.n+1):
            if i in solution and len_solution > 2: neighs.append(solution - {i})
            else: neighs.append(solution | {i})
        return neighs

    def get_candidate_nodes(self, solution):
        return [ n for n in self._nodes if n not in solution ]

    # def get_neighs_fast(self, solution):
    #     neighs = []
    #     M = len(solution[1])
    #     if len(self._W) == 0:
    #         bool_arr = [ i in solution[1] for i in range(0, self.n+1) ]
    #         self._W = [ self.W_np[i, bool_arr].sum() for i in range(1, self.n+1) ]

    #     for i in range(1,self.n+1):
    #         if i in solution[1] and M > 2:
    #             new_solution = solution[1] - {i}
    #             delta = (solution[0]-self._W[i-1])/(M-1)
    #         else:
    #             new_solution = solution[1] | {i}
    #             delta = (-solution[0]+self._W[i-1])/(M+1)

    #         neighs.append((delta, new_solution))

    #     for j,p in enumerate(self._W):
    #         bool_arr = [ i in new_solution for i in range(0, self.n+1) ]
    #         if (j+1) in new_solution:
    #             self._W[j] = p - self.W_np[j,bool_arr].sum()
    #         else:
    #             self._W[j] = p + self.W_np[j,bool_arr].sum()
    #     return neighs

    def get_random_solution(self):
        return set(random.sample(self._nodes, random.randint(2, self.n)))

