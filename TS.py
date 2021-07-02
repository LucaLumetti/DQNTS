from MMDP import MMDP
import time
from functools import lru_cache

class TabuSearch:
    def __init__(self, P, time_limit=100, alfa_limit=100):
        self.alfa_limit = alfa_limit
        self.time_limit = time_limit
        self._P = P
        self.TL = set()
        self.best_x = (0, {})
        self._sol_expolored = 0
        self.div_function = self.diversification

    def solve(self, div_function):
        self.__init__(self._P, self.time_limit, self.alfa_limit)
        self.start_time = time.time()
        while time.time() - self.start_time < self.time_limit:
            x = div_function()

            best, steps = self.intensification(frozenset(x[1]))
            self._sol_expolored += 1
            if best[0] > self.best_x[0]:
                 self.best_x = best
            print(f"steps: {steps}, best: {best[0]}")

        print(f"Explored: {self._sol_expolored}")
        return self.best_x

    def diversification(self):
        random_sol = self._P.get_random_solution()
        return (self._P.objective(random_sol), random_sol)

    def set_solution(self, solution):
        self.x = (self._P.objective(solution), solution)
        return

    @lru_cache(None)
    def intensification(self, raw_solution=None):
        if raw_solution is None: solution = self.x
        else: solution = (self._P.objective(raw_solution), raw_solution)

        alfa = 0
        steps = 0
        best = solution

        while alfa <= self.alfa_limit:
            # neighs = self._P.get_neighs(solution[1])
            # neighs = [ (self._P.objective(s), s) for s in neighs if s not in TL ]
            neighs = self._P.get_neighs_fast(solution)
            neighs = [ s for s in neighs if s[1] not in self.TL ]

            if len(neighs) == 0: return None
            time_diff = time.time() - self.start_time
            if time_diff > self.time_limit: break

            solution = max(neighs)
            alfa += 1

            steps += 1
            if best[0] < solution[0]:
                best = solution
                alfa = 0

            self.TL.add(frozenset(solution[1]))
        return best, steps
