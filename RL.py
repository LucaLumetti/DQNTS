import numpy as np
import torch
import random
import math
from collections import namedtuple
import os
import time
import glob
import datetime

from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from scipy.signal import medfilt
from functools import lru_cache, wraps
from MMDP import MMDP
from TS import TabuSearch

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

State = namedtuple('State', ('graph', 'partial_solution'))
Experience = namedtuple('Experience', ('state', 'state_tsr', 'action', 'reward', 'next_state', 'next_state_tsr'))

class QNet(nn.Module):
    """ The neural net that will parameterize the function Q(s, a)

        The input is the state (containing the graph and visited nodes),
        and the output is a vector of size N containing Q(s, a) for each of the N actions a.
    """

    def __init__(self, emb_dim, T=4):
        """ emb_dim: embedding dimension p
            T: number of iterations for the graph embedding
        """
        super(QNet, self).__init__()
        self.emb_dim = emb_dim
        self.T = T

        self.node_dim = emb_dim

        # We can have an extra layer after theta_1
        nr_extra_layers_1 = 2

        # Build the learnable affine maps:
        self.theta1 = nn.Linear(self.node_dim, self.emb_dim, True).to(device)
        self.theta2 = nn.Linear(self.emb_dim, self.emb_dim, True).to(device)
        self.theta3 = nn.Linear(self.emb_dim, self.emb_dim, True).to(device)
        self.theta4 = nn.Linear(1, self.emb_dim, True).to(device)
        self.theta5 = nn.Linear(2*self.emb_dim, 1, True).to(device)
        self.theta6 = nn.Linear(self.emb_dim, self.emb_dim, True).to(device)
        self.theta7 = nn.Linear(self.emb_dim, self.emb_dim, True).to(device)

        self.theta1_extras = [nn.Linear(self.emb_dim, self.emb_dim, True).to(device) for _ in range(nr_extra_layers_1)]

    def forward(self, xv, Ws):
        # xv: The node features (batch_size, num_nodes, node_dim)
        # Ws: The graphs (batch_size, num_nodes, num_nodes)

        num_nodes = xv.shape[1]
        batch_size = xv.shape[0]

        # print(f"xv.device: {xv.device}")
        # print(f"Ws.device: {Ws.device}")

        # pre-compute 1-0 connection matrices masks (batch_size, num_nodes, num_nodes)
        conn_matrices = torch.where(Ws > 0, torch.ones_like(Ws), torch.zeros_like(Ws)).to(device)

        # Graph embedding
        # Note: we first compute s1 and s3 once, as they are not dependent on mu
        mu = torch.zeros(batch_size, num_nodes, self.emb_dim, device=device)
        s1 = self.theta1(xv)  # (batch_size, num_nodes, emb_dim)
        for layer in self.theta1_extras:
            frelu = F.relu(s1)
            s1 = layer(F.relu(s1))  # we apply the extra layer

        s3_1 = F.relu(self.theta4(Ws.unsqueeze(3)))  # (batch_size, nr_nodes, nr_nodes, emb_dim) - each "weigth" is a p-dim vector
        s3_2 = torch.sum(s3_1, dim=1)  # (batch_size, nr_nodes, emb_dim) - the embedding for each node
        s3 = self.theta3(s3_2)  # (batch_size, nr_nodes, emb_dim)

        for t in range(self.T):
            s2 = self.theta2(conn_matrices.matmul(mu))
            mu = F.relu(s1 + s2 + s3)

        """ prediction
        """
        # we repeat the global state (summed over nodes) for each node,
        # in order to concatenate it to local states later
        global_state = self.theta6(torch.sum(mu, dim=1, keepdim=True).repeat(1, num_nodes, 1))

        local_action = self.theta7(mu)  # (batch_dim, nr_nodes, emb_dim)

        out = F.relu(torch.cat([global_state, local_action], dim=2))
        out = self.theta5(out).squeeze(dim=2)
        return out

class QFunction():
    def __init__(self, model, optimizer, lr_scheduler):
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loss_fn = nn.MSELoss()

    def predict(self, state_tsr, W):
        state_tsr = state_tsr.float().to(device)
        W = W.float().to(device)

        # print(f"state_tsr.device: {state_tsr.device}")
        # print(f"W.device: {W.device}")

        # batch of 1 - only called at inference time
        with torch.no_grad():
            estimated_rewards = self.model(state_tsr.unsqueeze(0), W.unsqueeze(0))
        return estimated_rewards[0]

    def evaluate_nodes(self, state_tsr, state):
        graph = state.graph
        W = graph.W.to(device)
        state_tsr = state_tsr.to(device)
        # print(W.device)
        # print(state_tsr.device)
        est_rewards = self.predict(state_tsr, W)  # size (nr_nodes,)
        est_rewards = np.interp(est_rewards.to("cpu"), (est_rewards.min(), est_rewards.max()), (-1, +1))
        idx = torch.arange(len(est_rewards))

        est_rewards = np.delete(est_rewards, state.partial_solution)
        idx = np.delete(idx, state.partial_solution)

        return idx, est_rewards

    def get_best_action(self, state_tsr, state):
        """ Computes the best (greedy) action to take from a given state
            Returns a tuple containing the ID of the next node and the corresponding estimated reward
        """
        graph = state.graph
        W = graph.W.to(device)
        state_tsr = state_tsr.to(device)
        # print(W.device)
        # print(state_tsr.device)
        estimated_rewards = self.predict(state_tsr, W)  # size (nr_nodes,)
        estimated_rewards = np.interp(estimated_rewards.to("cpu"), (estimated_rewards.min(), estimated_rewards.max()), (-1, +1))
        estimated_rewards = torch.from_numpy(estimated_rewards)
        sorted_reward_idx = estimated_rewards.argsort(descending=True)

        solution = state.partial_solution

        candidate_nodes = graph.get_candidate_nodes(solution)

        for idx in sorted_reward_idx.tolist():
            if idx in candidate_nodes:
                return int(idx), estimated_rewards[idx].item()
        return 0,0

    def batch_update(self, states_tsrs, Ws, actions, targets):
        """ Take a gradient step using the loss computed on a batch of (states, Ws, actions, targets)

            states_tsrs: list of (single) state tensors
            Ws: list of W tensors
            actions: list of actions taken
            targets: list of targets (resulting estimated rewards after taking the actions)
        """
        Ws_tsr = torch.stack(Ws).to(device)
        xv = torch.stack(states_tsrs).to(device)
        self.optimizer.zero_grad()

        # the rewards estimated by Q for the given actions
        # print(f"xv.device: {xv.device}")
        # print(f"Ws_tsr.device: {Ws_tsr.device}")
        estimated_rewards = self.model(xv.float(), Ws_tsr.float())
        estimated_rewards = estimated_rewards[range(len(actions)), actions]
        targets = torch.tensor(targets, dtype=torch.float32, device=device)

        # print(f"estimated: {estimated_rewards}")
        # print(f"targets: {targets}")

        loss = self.loss_fn(estimated_rewards, targets)
        loss_val = loss.item()

        loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step()

        return loss_val

class Memory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.nr_inserts = 0

    def remember(self, experience):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = experience
        self.position = (self.position + 1) % self.capacity
        self.nr_inserts += 1

    def sample_batch(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return min(self.nr_inserts, self.capacity)

def state2tens(state):
    solution = set(state.partial_solution)
    graph = state.graph
    W = graph.W
    total_W = graph.W.sum()
    candidate_nodes = graph.get_candidate_nodes(solution)
    candidate_nodes = list(candidate_nodes)
    solution = list(solution)
    xv = [[
        (1 if i in solution else 0),
        torch.sum(W[i,:]),
        torch.sum(W[i,solution]),
        torch.sum(W[i,candidate_nodes])
        ] for i in range(graph.n)]
    return torch.tensor(xv, dtype=torch.float32, requires_grad=False, device=device)

def init_model(P, fname=None):
    """ Create a new model. If fname is defined, load the model from the specified file.
    """
    Q_net = QNet(P["EMBEDDING_DIMENSIONS"], T=P["EMBEDDING_ITERATIONS_T"]).to(device)
    optimizer = optim.Adam(Q_net.parameters(), lr=P["INIT_LR"])
    lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=P["LR_DECAY_RATE"])

    if fname is not None:
        checkpoint = torch.load(fname)
        Q_net.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

    Q_func = QFunction(Q_net, optimizer, lr_scheduler)
    return Q_func, Q_net, optimizer, lr_scheduler

def checkpoint_model(model, optimizer, lr_scheduler, loss,
                     episode, avg_weight, folder_name='./models2'):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    fname = os.path.join(folder_name, 'ep_{}'.format(episode))
    fname += '_weight_{}'.format(avg_weight)
    fname += '.tar'

    torch.save({
        'episode': episode,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
        'loss': loss,
        'avg_weight': avg_weight
    }, fname)


def _moving_avg(x, N=10):
    return np.convolve(np.array(x), np.ones((N,))/N, mode='valid')

def training(P):
    Q_func, Q_net, optimizer, lr_scheduler = init_model(P)

    # Create memory
    memory = Memory(P["MEMORY_CAPACITY"])

    # Storing metrics about training:
    found_solutions = dict()  # episode --> (coords, W, solution)
    losses = []
    path_weights = []
    len_solutions = []

    # keep track of median solution weight for model checkpointing
    current_min_med_weight = float('-inf')

    for episode in range(P["NR_EPISODES"]):
        # sample a new random graph
        # graph = Graph(n=P["NR_NODES"], nu=P["GRAPH_NU"])
        graph = MMDP()
        ts = TabuSearch(graph)

        graph.W = graph.W.to(device)
        W = torch.tensor(graph.W_np, dtype=torch.float64, requires_grad=False, device=device)

        # current partial solution - a list of node index
        solution = []

        # current state (tuple and tensor)
        current_state = State(partial_solution=solution, graph=graph)
        current_state_tsr = state2tens(current_state)

        # Keep track of some variables for insertion in replay memory:
        states = [current_state]
        states_tsrs = [current_state_tsr]  # we also keep the state tensors here (for efficiency)
        rewards = []
        est_rewards = []
        actions = []

        # current value of epsilon
        epsilon = max(P["MIN_EPSILON"], (1-P["EPSILON_DECAY_RATE"])**episode)
        # epsilon = 0.9

        nr_explores = 0
        t = -1
        is_solution_final = False

        # construct the initial solution
        while True:
            candidate_nodes = graph.get_candidate_nodes(solution)
            if len(candidate_nodes) == 0: break

            if epsilon >= random.random(): # pick a random solution
                next_node = random.sample(candidate_nodes, 1)[0]
                est_rewards.append(None)
            else: # select best node
                next_node, est_reward = Q_func.get_best_action(current_state_tsr, current_state)
                if est_reward < 0 and len(solution) > 2: break
                est_rewards.append(est_reward)

            next_solution = solution + [next_node]

            next_state = State(partial_solution=next_solution, graph=graph)
            next_state_tsr = state2tens(next_state)

            states.append(next_state)
            states_tsrs.append(next_state_tsr)
            actions.append(next_node)

            current_state = next_state
            current_state_tsr = next_state_tsr
            solution = next_solution

        len_solutions.append(len(solution))
        # store rewards and states obtained along this episode:
        best, _ = ts.intensification(frozenset(solution))
        best_objective = best[0]
        best_solution = best[1]
        # not_in = sum([ 1 for node in solution if node not in best_solution ])
        # in_best = len(best_solution)

        rewards = [ 1 if node in best_solution else -1 for node in solution ]


        # print(f"solution: {solution}")
        # print(f"best_solution: {best_solution}")
        # print(f"actions: {actions}")
        # print(f"rewards: {rewards}")
        # print(f"est_rewards: {est_rewards}")

        for n in range(1, len(solution)):
            memory.remember(Experience(state=states[-n],
                                    state_tsr=states_tsrs[-n],
                                    action=actions[-n],
                                    reward=sum(rewards[-n:]),
                                    next_state=next_state,
                                    next_state_tsr=next_state_tsr))
        # take a gradient step
        loss = None
        if len(memory) >= P["BATCH_SIZE"]:
            experiences = memory.sample_batch(P["BATCH_SIZE"])

            batch_states_tsrs = [e.state_tsr for e in experiences]
            batch_Ws = [e.state.graph.W for e in experiences]
            batch_actions = [e.action for e in experiences]
            batch_targets = []

            for i, experience in enumerate(experiences):
                target = experience.reward
                _, best_reward = Q_func.get_best_action(experience.next_state_tsr, experience.next_state)
                if best_reward >= 0:
                    target += P["GAMMA"] * best_reward
                batch_targets.append(target)

            # print('batch targets: {}'.format(batch_targets))
            loss = Q_func.batch_update(batch_states_tsrs, batch_Ws, batch_actions, batch_targets)
            losses.append(loss)

            med_weight = np.median(path_weights[-100:])
            if med_weight > current_min_med_weight:
                current_min_med_weight = med_weight
                checkpoint_model(Q_net, optimizer, lr_scheduler, loss, episode, med_weight, P["FOLDER_NAME"])

        weight = sum(rewards)
        path_weights.append(weight)

        if episode % 10 == 0:
            print('Ep %d. Loss = %.3f / median weight = %.3f / last = %.4f / epsilon = %.4f / lr = %.4f / memory = %d / sol_length = %d' % (
                episode,
                (-1 if loss is None else loss),
                np.median(path_weights[-10:]),
                weight,
                epsilon,
                Q_func.optimizer.param_groups[0]['lr'],
                len(memory),
                np.mean(len_solutions[-10:])
                )
            )





        # while True:
        #     t += 1  # time step of this episode

        #     est_reward = None
        #     if len(solution) > graph.n*3//4: break
        #     if epsilon >= random.random():
        #         # explore
        #         candidate_nodes = graph.get_candidate_nodes(solution)
        #         if len(candidate_nodes) == 0:
        #             break
        #         next_node = random.sample(candidate_nodes, 1)[0]
        #         nr_explores += 1
        #     else:
        #         # exploit
        #         next_node, est_reward = Q_func.get_best_action(current_state_tsr, current_state)
        #         if est_reward < 0:
        #             for n in range(1, len(solution)):
        #                 memory.remember(Experience(state=states[-n],
        #                                         state_tsr=states_tsrs[-n],
        #                                         action=actions[-n],
        #                                         reward=sum(rewards[-n:]),
        #                                         next_state=next_state,
        #                                         next_state_tsr=next_state_tsr))
        #             break

        #     next_solution = solution + [next_node]

        #     # reward observed for taking this step
        #     # _, steps_next_solution = ts.intensification(frozenset(next_solution))
        #     # _, steps_solution = ts.intensification(frozenset(solution))
        #     reward = graph.objective(next_solution) - graph.objective(solution)

        #     if episode % 50 == 0 and est_reward is not None:
        #         print('Ep {} | current sol: {} / reward: {} / next est reward: {}'.format(episode, solution, reward, est_reward))

        #     # print(f"{steps_solution} -> {steps_next_solution}: {reward}")

        #     next_state = State(partial_solution=next_solution, graph=graph)
        #     next_state_tsr = state2tens(next_state)

        #     # store rewards and states obtained along this episode:
        #     states.append(next_state)
        #     states_tsrs.append(next_state_tsr)
        #     rewards.append(reward)
        #     actions.append(next_node)

        #     # store our experience in memory, using n-step Q-learning:
        #     # if len(solution) >= N_STEP_QL:
        #     if len(solution) >= P["N_STEP_QL"]:
        #         memory.remember(Experience(state=states[-P["N_STEP_QL"]],
        #                                 state_tsr=states_tsrs[-P["N_STEP_QL"]],
        #                                 action=actions[-P["N_STEP_QL"]],
        #                                 reward=sum(rewards[-P["N_STEP_QL"]:]),
        #                                 next_state=next_state,
        #                                 next_state_tsr=next_state_tsr))

        #     # if graph.is_solution_final(tuple(solution)):
        #     #     for n in range(1, len(solution)):
        #     #         memory.remember(Experience(state=states[-n],
        #     #                                 state_tsr=states_tsrs[-n],
        #     #                                 action=actions[-n],
        #     #                                 reward=sum(rewards[-n:]),
        #     #                                 next_state=next_state,
        #     #                                 next_state_tsr=next_state_tsr))

        #     # update state and current solution
        #     current_state = next_state
        #     current_state_tsr = next_state_tsr
        #     solution = next_solution

        #     # take a gradient step
        #     loss = None
        #     if len(memory) >= P["BATCH_SIZE"]:
        #         experiences = memory.sample_batch(P["BATCH_SIZE"])

        #         batch_states_tsrs = [e.state_tsr for e in experiences]
        #         batch_Ws = [e.state.graph.W for e in experiences]
        #         batch_actions = [e.action for e in experiences]
        #         batch_targets = []

        #         for i, experience in enumerate(experiences):
        #             target = experience.reward
        #             _, best_reward = Q_func.get_best_action(experience.next_state_tsr, experience.next_state)
        #             if best_reward >= 0:
        #                 target += P["GAMMA"] * best_reward
        #             batch_targets.append(target)

        #         # print('batch targets: {}'.format(batch_targets))
        #         loss = Q_func.batch_update(batch_states_tsrs, batch_Ws, batch_actions, batch_targets)
        #         losses.append(loss)

        #         """ Save model when we reach a new low average solution weight
        #         """
        #         med_weight = np.median(path_weights[-100:])
        #         if med_weight > current_min_med_weight:
        #             current_min_med_weight = med_weight
        #             checkpoint_model(Q_net, optimizer, lr_scheduler, loss, episode, med_weight)

        # # length = total_distance(solution, W)
        # # print(f"solution ({graph.objective(solution)}): {solution}")
        # weight = int(graph.objective(solution))
        # # _, greedy_weight = graph.greedy_co()
        # path_weights.append(weight)

        # if episode % 10 == 0:
        #     print('Ep %d. Loss = %.3f / median weight = %.3f / last = %.4f / epsilon = %.4f / lr = %.4f / memory = %d / sol_length = %d' % (
        #         episode, (-1 if loss is None else loss), np.median(path_weights[-50:]), weight, epsilon,
        #         Q_func.optimizer.param_groups[0]['lr'], len(memory), len(solution)))
        #     # found_solutions[episode] = (W.clone(), graph.coords.copy(), [n for n in solution])

    # plot loss and obj
    plt.figure(figsize=(8,5))
    plt.title("Loss function")
    plt.semilogy(_moving_avg(losses, 100))
    plt.ylabel('loss')
    plt.xlabel('training iteration')
    plt.savefig(f'nn_loss.png')

    plt.figure(figsize=(8,5))
    plt.title("Objective function")
    plt.plot(_moving_avg(path_weights, 100))
    plt.ylabel('average objective')
    plt.xlabel('episode')
    plt.savefig(f'nn_obj.png')

def test(P, instance):
    all_weights_fnames = [f for f in os.listdir(P["FOLDER_NAME"]) if f.endswith('.tar')]
    highest_fname = sorted(all_weights_fnames, key=lambda s: float(s.split('.tar')[0].split('_')[-1]))[-1]
    # print('max avg weight found: {}'.format(highest_fname.split('.tar')[0].split('_')[-1]))

    """ Load checkpoint
    """
    Q_func, Q_net, optimizer, lr_scheduler = init_model(P)

    """ A function to plot solutions
    """

    """ Generate example solutions
    """

    total_objective_nn = []
    total_objective_greedy = []
    total_time_nn = []
    total_time_greedy = []

    graph = MMDP(instance)
    time_limit = 10
    if "500" in instance or "1000" in instance or "750" in instance:
        time_limit = 100
    ts = TabuSearch(graph, time_limit=time_limit)

    print(f"Testing over {instance} at {datetime.datetime.now()} for {time_limit}s")

    W = torch.tensor(graph.W_np, dtype=torch.float32, requires_grad=False, device=device)


    # generate a initial solution
    # def diversification():
    #     nn_solution = []
    #     current_state = State(partial_solution=nn_solution, graph=graph)
    #     current_state_tsr = state2tens(current_state)
    #     while True:
    #         next_node, est_reward = Q_func.get_best_action(current_state_tsr, current_state)
    #         if est_reward < 0: break
    #         nn_solution = nn_solution + next_node
    #         current_state = State(partial_solution=nn_solution, graph=graph)
    #         current_state_tsr = state2tens(current_state)
    #     print(f"NN_SOL: {nn_solution}")
    #     return (graph.objective(nn_solution), nn_solution)
    def diversification():
        nn_solution = []
        current_state = State(partial_solution=nn_solution, graph=graph)
        current_state_tsr = state2tens(current_state)
        idx, est_rewards = Q_func.evaluate_nodes(current_state_tsr,  current_state)

        good_nodes = idx[est_rewards >= 0]
        min_nodes = max(len(good_nodes)//4, 2)
        selected_node = random.sample(list(good_nodes), random.randint(min_nodes, len(good_nodes)))
        selected_node = [ int(n) for n in selected_node ]
        nn_solution += selected_node

        current_state = State(partial_solution=nn_solution, graph=graph)
        current_state_tsr = state2tens(current_state)
        # while True:
        #     idx, est_rewards = Q_func.evaluate_nodes(current_state_tsr,  current_state)
        #     # if (est_rewards >= 0).sum() == 0 and len(nn_solution) >= 2: break
        #     if len(nn_solution) >= 2: break

        #     good_nodes = idx[est_rewards >= 0]
        #     selected_node = random.sample(list(good_nodes), random.randint(2, len(good_nodes)))
        #     nn_solution += selected_node

        #     current_state = State(partial_solution=nn_solution, graph=graph)
        #     current_state_tsr = state2tens(current_state)
        nn_solution = list(nn_solution)
        # print(f"NN_SOL: {len(nn_solution)}")
        return (graph.objective(nn_solution), nn_solution)

    f = open("results.txt", "a")
    r = open(f"solutions/{instance}.sol", "a")

    solution = ts.solve(diversification)
    print(f"DRL result: {solution[0]}")

    f.write(f"{instance}: {solution[0]}\n")
    r.write(f"{solution[1]}")

def main(parameters):
    training(parameters)
    # typeI = glob.glob("instances/typeI/*")
    # typeII = glob.glob("instances/typeII/*")
    # instances = typeI + typeII
    # instances.sort()
    # for instance in instances:
    #     test(parameters, instance)

if __name__ == "__main__":
    PARAMETERS={
            "SEED": 1,

            # Graph
            # "NR_NODES": 40,
            # "GRAPH_NU": 25
            "EMBEDDING_DIMENSIONS": 4,
            "EMBEDDING_ITERATIONS_T": 5,

            # Learning
            "NR_EPISODES": 5001,
            "MEMORY_CAPACITY": 2048,
            "N_STEP_QL": 4,
            "BATCH_SIZE": 64,
            "GAMMA": 0.5,
            "INIT_LR": 5e-3,
            "LR_DECAY_RATE": 1. - 5e-5,
            "MIN_EPSILON": 0.25,
            # "EPSILON_DECAY_RATE": 2e-3,
            "EPSILON_DECAY_RATE": 5e-5,

            # where to save best models
            "FOLDER_NAME": './models'
            }

    torch.manual_seed(PARAMETERS["SEED"])
    np.random.seed(PARAMETERS["SEED"])
    random.seed(PARAMETERS["SEED"])

    main(PARAMETERS)
