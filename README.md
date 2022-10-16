# DQNTS: Deep Q-Learning Tabu Search
### Introduction
In this project, I've tried to face the Max-Mean Dispersion Problem using Tabu Search guided by a deep reinforcement learning algorithm during the dispersion phase.
Given a complete graph G(V, E) where each edge has associated a distance or affinity $d_{ij}$, the Max Mean Dispersion Problem is the search of a subset of vertex $M \subset V, |M| >= 2$ which maximize the mean dispersion, calculated as follow:

$$
MeanDispersion(M) = \frac{\sum_{i,j \in M}d_{ij}}{|M|}
$$

This problem is known to be strongly NP-hard and has the characteristic that the subset $M$ does not have a fixed size, but can vary between $2$ and $|V|$.

### RLTS
In 2020, Nijimbere et al. proposed an approach based on the
combination of reinforcement learning and tabu search, named RLTS. The main idea is to use Q-Learning to build a high-quality initial solution, then improving it with a one-flip tabu search. Then Q-Learning algorithm is trained during the search of the solution, by giving a positive or negative reward if the node remain or not in the solution after the tabu search.

### DQNTS
My proposed solution is to use a Deep Q-Network to generate the initial solution and tabu search to improve it, hence the name DQNTS. The approach is similar to \cite{nijimbere2020tabu}, the network learns a heuristic to build the initial solution, then uses a one-flip tabu search to improve that solution. The difference is that the DQN is pre-trained, hence there is no need to start with a random solution and I can generate a high-quality initial solution from the beginning.

### Network Architecture
The network architecture is based on "Learning combinatorial optimization algorithms over graphs" by Dai et al.
The state2tens embedding is done with 4 features extracted from each node, which are:
- 1 if the node in the solution, 0 otherwise
- the sum of all edges connected to the node
- the sum of all edges connected to the node and the solution nodes
- the sum of all edges connected to the node and the nodes not in the solution

### Network training
To train the network, first, we construct a feasible solution using an $\epsilon$-greedy strategy, stopping when no positive rewards are predicted by the network.  Then the solution is given to the one-flip tabu search which
improves it and returns the best solution. For every node in the initial solution which remains in the final solution after tabu search, a reward of +1 is given, otherwise, the reward is -1. The network has been trained over 10 different instances (MDPIx\_35 and MDPIIx\_35, $1 <= x <= 10$) for 5001 episodes.

### Tabu Search
The tabu search implementation is the same as in \cite{nijimbere2020tabu}, with the only difference in the parameter $\alpha = 100$ instead of $\alpha = 50000$. My implementation couldn't finish even a single iteration in the time limit imposed with the parameter $\alpha$ proposed in the paper. This made me think that my implementation is way slower, but still, I left the same time constraints as the results were good enough.

### General Algorithm
DQN is used during the construction of the initial solution:
for all the nodes, the network estimates the reward, then all the values estimated get interpolated in the range $[-1,+1]$. All the nodes with a value $>= 0$ are named as "good nodes". Among these "good nodes", a random amount is taken to construct the initial solution. Picking one node at a time would result in a better solution, but this approach was too slow for large graphs. This solution is then processed with a one-flip tabu search until no best solutions are found for $\alpha = 100$ iterations in a row.  Now a new initial solution is generated and the process is repeated until the time limit is not violated. Finally, the best solution found is returned.

### Result
Results can be seen on the pdf [presentation](https://github.com/LucaLumetti/DQNTS/tree/main/pdf) uploaded in this repository under the pdf directory.
