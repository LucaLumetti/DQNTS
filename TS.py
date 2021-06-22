from MMDP import MMDP
import time

class TabuSearch:
    def __init__(self, P, time_limit=100, alfa_limit=10):
        self.alfa_limit = alfa_limit
        self.time_limit = time_limit
        self._P = P
        self.best_x = (0, {})

    def solve(self):
        self.start_time = time.time()
        while time.time() - self.start_time < self.time_limit:
            self.diversification()
            self.intensification()
            print(self.best_x[0])
        return self.best_x

    def diversification(self):
        random_sol = self._P.get_random_solution()
        self.x = (self._P.objective(random_sol), random_sol)
        return

    def intensification(self):
        alfa = 0
        best = self.x
        TL = set()

        while alfa <= self.alfa_limit:
            neighs = self._P.get_neighs(self.x[1])
            neighs = [ (self._P.np_objective(s), s) for s in neighs if s not in TL ]
            # __neighs = self._P.get_neighs_fast(self.x)

            if len(neighs) == 0: break
            if time.time() - self.start_time > self.time_limit: break

            self.x = max(neighs)
            alfa += 1

            if best[0] < self.x[0]:
                best = self.x
                alfa = 0

            TL.add(frozenset(self.x[1]))

        if best[0] > self.best_x[0]:
            self.best_x = best
        return


mmdp = MMDP("./instances/typeI/MDPI1_150.txt")
ts = TabuSearch(mmdp)
solution = ts.solve()
print(solution)
