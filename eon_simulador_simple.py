import simpy
from config import *
from random import *
import numpy as np
import networkx as nx
import math
import utils
from collections import defaultdict
from tqdm import tqdm


topology = nx.read_weighted_edgelist('topology/' + TOPOLOGY, nodetype=int)
topology = topology.to_directed()


class Desalocate(object):
    def __init__(self, env):
        self.env = env
    def Run(self, count, path, spectro, holding_time):
        global topology
        yield self.env.timeout(holding_time)
        for i in range(0, (len(path)-1)):
            for slot in range(spectro[0],spectro[1]+1):
                topology[path[i]][path[i+1]]['capacity'][slot] = 0


class QLearningRouter:
    def __init__(self, topology, epsilon=0.9, alpha=0.1, gamma=0.9, max_episodes=500):
        self.topology = topology
        self.num_nodes = len(topology.nodes())
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.max_episodes = max_episodes
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.feasible_paths = []
        self.random = Random()

    def calculate_link_fragmentation(self, node, next_node):
        """
        Calculate entropy-based fragmentation metric for a link
        Based on equation (2) from the paper
        """
        link_capacity = self.topology[node][next_node]['capacity']
        S = len(link_capacity)  # Total number of slots
        
        # Find contiguous free fragments
        fragments = []
        current_fragment = 0
        
        for slot in link_capacity:
            if slot == 0:  # Free slot
                current_fragment += 1
            elif current_fragment > 0:  # End of fragment
                fragments.append(current_fragment)
                current_fragment = 0
                
        # Add last fragment if exists
        if current_fragment > 0:
            fragments.append(current_fragment)
            
        if not fragments:  # No free fragments
            return float('inf')
            
        # Calculate entropy-based fragmentation
        fragmentation = 0
        for fragment_size in fragments:
            ratio = fragment_size / S
            fragmentation += ratio * math.log(S / fragment_size)
            
        return fragmentation

    def compute_link_availability(self, node, next_node):
        """
        Calculate link availability as ratio of free slots
        """
        link_capacity = self.topology[node][next_node]['capacity']
        free_slots = sum(1 for slot in link_capacity if slot == 0)
        total_slots = len(link_capacity)
        return free_slots / total_slots

    def get_neighbors(self, node):
        return list(self.topology.neighbors(node))

    def get_possible_actions(self, current_node, destination):
        return [(next_node, destination) for next_node in self.get_neighbors(current_node)]

    def calculate_reward(self, current_node, next_node, visited):
        """
        Calculate reward based on link fragmentation and availability
        """
        if next_node in visited:
            return -0.5
            
        # Calculate link fragmentation and availability
        link_fragmentation = self.calculate_link_fragmentation(current_node, next_node)
        link_availability = self.compute_link_availability(current_node, next_node)
        
        # Higher availability and lower fragmentation are preferred
        reward = link_availability - link_fragmentation
        
        return reward

    def find_routes(self, source, destination, k=3):
        """Find k routes from source to destination"""
        self.feasible_paths = []
        epsilon = self.epsilon

        for episode in range(self.max_episodes):
            visited = []
            current_node = source
            path = [current_node]
            
            # Decay epsilon
            epsilon = max(0.01, epsilon * 0.9999)
            
            while current_node != destination:
                possible_actions = self.get_possible_actions(current_node, destination)
                if not possible_actions:
                    break

                # Epsilon-greedy action selection
                if self.random.random() < epsilon:
                    next_node, _ = self.random.choice(possible_actions)
                else:
                    next_node = max(possible_actions, 
                                  key=lambda x: self.q_table[current_node][x])[0]

                if next_node in visited or len(path) > self.num_nodes:
                    break

                # Calculate reward based on fragmentation and availability
                reward = self.calculate_reward(current_node, next_node, visited)
                
                # Update Q-value
                old_q = self.q_table[current_node][(next_node, destination)]
                next_actions = self.get_possible_actions(next_node, destination)
                max_next_q = max([self.q_table[next_node][action] 
                                for action in next_actions]) if next_actions else 0
                
                new_q = (1 - self.alpha) * old_q + self.alpha * (reward + self.gamma * max_next_q)
                self.q_table[current_node][(next_node, destination)] = new_q

                visited.append(current_node)
                current_node = next_node
                path.append(current_node)

            if current_node == destination and path not in self.feasible_paths:
                self.feasible_paths.append(path)

        return self._get_k_best_paths(source, destination, k)

    def _get_k_best_paths(self, source, destination, k):
        """Get k best paths based on cumulative Q-values"""
        path_scores = []
        for path in self.feasible_paths:
            score = 0
            for i in range(len(path)-1):
                score += self.q_table[path[i]][(path[i+1], destination)]
            path_scores.append((path, score))

        path_scores.sort(key=lambda x: x[1], reverse=True)
        return [path for path, _ in path_scores[:k]]

class Simulador(object):
    def __init__(self, env):
        self.env = env
        global topology
        for u, v in list(topology.edges):
            topology[u][v]['capacity'] = [0] * SLOTS
        self.nodes = list(topology.nodes())
        self.random = Random()
        self.NumReqBlocked = 0
        self.cont_req = 0
        self.connModulationInfo = {}
        # Initialize Q-Learning router
        self.router = QLearningRouter(topology)

    def Run(self, rate):
        global topology
        
        #for count in range(1, NUM_OF_REQUESTS + 1):
        for count in tqdm(range(1, NUM_OF_REQUESTS+1), desc="Processing Requests"):
            yield self.env.timeout(self.random.expovariate(rate))
            src, dst = self.random.sample(self.nodes, 2)
            bandwidth = int(self.random.uniform(1, 200))
            holding_time = self.random.expovariate(HOLDING_TIME)
            
            # Use Q-Learning to find routes
            paths = self.router.find_routes(src, dst, N_PATH)
            flag = 0
            
            for path in paths:
                distance = int(utils.Distance(topology, path))
                if distance <= 4000:
                    num_slots, m = utils.Modulation(distance, bandwidth)
                    self.check_path = utils.PathIsAble(num_slots, path, topology)
                    
                    if self.check_path[0] == True:
                        OSNR_r_path = utils.computeOSNR(topology, path, num_slots, self.check_path[1], m, self.connModulationInfo)
                        #print("m", m, "OSNR_r_path",OSNR_r_path)
                        check_OSNR_Th= utils.check_OSNR_Th(m, OSNR_r_path)
                        if check_OSNR_Th == True:
                            self.cont_req += 1
                            OSNR_reward = 1
                            self.update_q_values_for_path(path, OSNR_reward)
                            utils.FirstFit(topology,count, self.check_path[1],self.check_path[2],path)
                            self.connModulationInfo[count] = m
                            spectro = [self.check_path[1], self.check_path[2]]
                            desalocate = Desalocate(self.env)
                            self.env.process(desalocate.Run(count,path,spectro,holding_time))
                            flag=1
                            break
                        else:
                            OSNR_reward = -1
                            self.update_q_values_for_path(path, OSNR_reward)
                    
            if flag == 0:
                self.NumReqBlocked += 1

    def update_q_values_for_path(self, path, osnr):
        """Update Q-values for successful path"""
        # For each link in the path
        for i in range(len(path)-1):
            current = path[i]
            next_node = path[i+1]
            destination = path[-1]
            
            # Calculate link metrics
            link_availability = self.router.compute_link_availability(current, next_node)
            link_fragmentation = self.router.calculate_link_fragmentation(current, next_node)
            
            # Calculate reward for successful path
            reward = osnr + link_availability - link_fragmentation
            
            # Calculate max future Q-value
            if i < len(path) - 2:  # If not the last link
                future_actions = self.router.get_possible_actions(next_node, destination)
                max_future_q = max([self.router.q_table[next_node][action] 
                                  for action in future_actions]) if future_actions else 0
            else:
                max_future_q = 0
            
            # Update Q-value using Q-learning update rule
            old_q = self.router.q_table[current][(next_node, destination)]
            new_q = (1 - self.router.alpha) * old_q + \
                    self.router.alpha * (reward + self.router.gamma * max_future_q)
            self.router.q_table[current][(next_node, destination)] = new_q
