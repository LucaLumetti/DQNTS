import pandas as pd
import random
import numpy as np
import time
import torch
GRAPH_SIZE = 35

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class MMDP:
    def __init__(self, input_file=None):
        files = [ f"./instances/typeI/MMDPI{n}_{GRAPH_SIZE}.txt" for n in range(1,11) ]
        files += [ f"./instances/typeII/MMDPII{n}_{GRAPH_SIZE}.txt" for n in range(1,11) ]

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
        self.W_np_full = np.zeros((self.n, self.n))

        for line in lines:
            split = line.strip().split("\t")
            split = [ int(split[0])-1, int(split[1])-1, float(split[2]) ]
            self.W_np[split[0], split[1]] = split[2]
            self.W_np_full[split[0], split[1]] = split[2]
            self.W_np_full[split[1], split[0]] = split[2]

        self.W = torch.from_numpy(self.W_np).to(device)
        self.W_full = torch.from_numpy(self.W_np_full).to(device)

    def objective(self, solution):
        len_sol = len(solution)
        if len_sol == 0: return 0
        s = torch.tensor(list(solution))
        idx = np.ix_(s, s)
        f = self.W[idx].sum()
        f /= len_sol
        return f

    # def get_neighs(self, solution):
    #     neighs = []
    #     len_solution = len(solution[1])
    #     for i in range(0,self.n):
    #         if i in solution and len_solution > 2: neighs.append(solution - {i})
    #         else: neighs.append(solution | {i})
    #     return neighs

    def get_candidate_nodes(self, solution):
        return [ n for n in self._nodes if n not in solution ]

    def get_neighs_fast(self, solution):
        neighs = []
        M = len(solution[1])
        sol_tens = torch.tensor(list(solution[1]))
        for i in range(0,self.n):
            new_obj = solution[0]
            if i in solution[1] and M > 2:
                new_solution = solution[1] - {i}
                new_obj = (new_obj*M - self.W_full[i,sol_tens].sum())/(M-1)
            else:
                new_solution = solution[1] | {i}
                new_obj = (new_obj*M + self.W_full[i,sol_tens].sum())/(M+1)

            neighs.append((new_obj, new_solution))
        return neighs

    def get_random_solution(self):
        return set(random.sample(self._nodes, random.randint(2, self.n)))

