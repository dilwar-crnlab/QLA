import simpy
from config import *
from random import *
import numpy as np
import networkx as nx
import math
import utils
from collections import defaultdict
from tqdm import tqdm
import logging
from typing import List, Dict, Tuple
from queue import PriorityQueue


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



class QStarRouter:
    def __init__(self, topology: nx.Graph, max_path_length: float = float('inf')):
        """
        Initialize Q* router for NetworkX topology
        
        Args:
            topology: NetworkX graph with weighted edges
            max_path_length: Maximum allowed path length
        """
        self.topology = topology
        self.MAX_LEN = max_path_length
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.global_link_fragmentation = {}
        self.global_link_availability = {}
        #self.path_cache = {}  # Cache for found paths
        
    def find_k_paths(self, src: int, dst: int, k: int) -> List[List[int]]:
        """
        Find k shortest paths using Q* search
        
        Args:
            src: Source node
            dst: Destination node
            k: Number of paths to find
            
        Returns:
            List of k paths (each path is a list of nodes)
        """
        # Check cache first
        # cache_key = (src, dst, k)
        # if cache_key in self.path_cache:
        #     return self.path_cache[cache_key]
            
        paths = []
        visited_paths = set()
        open_queue = PriorityQueue()
        
        # Initial state: (node, path_length, path_nodes)
        start_state = (src, 0.0, [src])
        
        # Add initial actions to queue
        self._add_initial_actions(start_state, dst, open_queue)
        
        while not open_queue.empty() and len(paths) < k:
            f_cost, (state, action) = open_queue.get()
            
            current_node, path_len, path_nodes = state
            next_node = action
            
            # Check length constraint
            edge_data = self.topology.get_edge_data(current_node, next_node)
            if not edge_data:
                continue
                
            next_length = path_len + edge_data['weight']
            if next_length > self.MAX_LEN:
                continue
                
            # Generate next state
            next_state = (
                next_node,
                next_length,
                path_nodes + [next_node]
            )
            
            # Check if destination reached
            if next_node == dst:
                path_key = tuple(next_state[2])
                if path_key not in visited_paths:
                    paths.append(next_state[2])
                    visited_paths.add(path_key)
                    
                    # Update Q-values with positive reward
                    #reward = self._calculate_reward(next_state)
                    self._update_q_values(state, action, next_state)
                continue
                
            # Get valid actions from next state
            valid_actions = self._get_valid_actions(next_state, dst)
            
            # Add next actions to queue
            for next_action in valid_actions:
                next_f_cost = self._calculate_f_cost(next_state, next_action)
                open_queue.put((next_f_cost, (next_state, next_action)))
            
            # Update Q-values
            #reward = self._calculate_reward(next_state)
            self._update_q_values(state, action, next_state)
        
        # Cache the results
        #self.path_cache[cache_key] = paths
        return paths
    
    def _calculate_f_cost(self, state, action) -> float:
        """Calculate f(s,a) = g(s) + Q(s,a)"""
        current_node, path_len, _ = state
        next_node = action
        
        # Get edge weight
        edge_data = self.topology.get_edge_data(current_node, next_node)
        if not edge_data:
            return float('inf')
            
        # f(s,a) = g(s) + Q(s,a)
        g_cost = path_len
        q_value = self.q_table[self._get_state_key(state)][action]
        
        return g_cost + q_value
    
    def calculate_link_fragmentation(self, link_capacity):
        S = len(link_capacity)  # Total number of spectrum slots
        free_fragments = []
        current_fragment = 0
        # Identify free fragments
        for slot in link_capacity:
            if slot == 0:
                current_fragment += 1
            else:
                if current_fragment > 0:
                    free_fragments.append(current_fragment)
                    current_fragment = 0
        if current_fragment > 0:
            free_fragments.append(current_fragment)
        # Calculate entropy-based fragmentation for this link
        link_fragmentation = 0
        for w in free_fragments:
            link_fragmentation += (w / S) * math.log(S / w)
        return link_fragmentation

    def compute_link_availability(self, link_slots):
        """
        Compute the link availability based on the number of available slots.
        
        :param link_slots: A list where each element represents a slot on the link.
                        0 indicates an available slot, and anything > 0 indicates an occupied slot.
        :return: Link availability as a ratio (value between 0 and 1).
        """
        total_slots = len(link_slots)
        available_slots = link_slots.count(0)  # Count slots that are available (0)
        # Calculate link availability as the ratio of available slots to total slots
        link_availability = available_slots / total_slots
        return link_availability

    def _calculate_immediate_reward(self, current_node: int, next_node: int) -> float:
        """
        Calculate immediate reward for taking a single action (moving to next_node)
        This considers only the current link metrics
        
        Returns:
            float: Immediate reward for taking this action
        """
        # Get link data
        edge_data = self.topology.get_edge_data(current_node, next_node)
        if not edge_data:
            return float('-inf')  # Invalid link

        capacity = edge_data.get('capacity', [])
        # 1. Link Length Component
        link_length = edge_data['weight']
        length_reward = (1/link_length)
        
        # 2. Current Link availability
        availability_reward = self.compute_link_availability(capacity)      
        # 3. Current Link Fragmentation
        fragmentation = self.calculate_link_fragmentation(capacity)
        # Combine immediate metrics
        immediate_reward = length_reward + availability_reward - fragmentation
        return immediate_reward

    def _calculate_future_reward(self, state) -> float:
        """
        Calculate future reward based on complete path properties
        This considers the entire path's characteristics
        
        Returns:
            float: Future reward based on path metrics
        """
        _, path_len, path_nodes = state
        
        if len(path_nodes) < 2:
            return 0.0

        # Accumulate path metrics
        total_length = 0.0
        total_avail = 0.0
        total_frag = 0.0
        num_links = len(path_nodes) - 1

        # Calculate metrics for entire path
        for i in range(num_links):
            u, v = path_nodes[i], path_nodes[i+1]
            edge_data = self.topology.get_edge_data(u, v)
            
            if edge_data and 'capacity' in edge_data:
                # 1. Accumulate Length
                total_length += edge_data['weight']
                
                # 2. Calculate Path Utilization
                capacity = edge_data['capacity']
                availability = self.compute_link_availability(capacity)
                total_avail += availability
                
                # 3. Calculate Path Fragmentation
                fragments = self.calculate_link_fragmentation(capacity)
                total_frag += fragments

        # Average metrics across path
        avg_length = total_length / num_links
        avg_avail = total_avail / num_links
        avg_frag = total_frag / num_links

        # # Calculate future rewards components
        length_reward = (1/avg_length)
        # utilization_reward = -self.weights['utilization'] * avg_util
        # fragmentation_reward = -self.weights['fragmentation'] * avg_frag

        # # Additional path-specific metrics
        # path_length_factor = -0.1 * len(path_nodes)  # Penalize very long paths
        

        # Combine future metrics
        future_reward = (length_reward + avg_avail - avg_frag)

        return future_reward

    def _update_q_values(self, state, action, next_state):
        """
        Update Q-values using both immediate and future rewards
        Q(s,a) = Q(s,a) + lr * [R_immediate + gamma * (R_future + max_a' Q(s',a')) - Q(s,a)]
        """
        current_node = state[0]
        next_node = action

        # 1. Calculate immediate reward for the action
        immediate_reward = self._calculate_immediate_reward(current_node, next_node)
        
        # 2. Calculate future reward for resulting state
        future_reward = self._calculate_future_reward(next_state)
        
        # 3. Get Q-value for best next action
        next_state_key = self._get_state_key(next_state)
        next_q = max(
            [self.q_table[next_state_key][a] 
             for a in self._get_valid_actions(next_state, None)],
            default=0
        )
        
        # 4. Q-learning update
        state_key = self._get_state_key(state)
        lr = 0.1  # Learning rate
        gamma = 0.99  # Discount factor
        current_q = self.q_table[state_key][action]
        
        # Combined update with both rewards
        self.q_table[state_key][action] = current_q + lr * (
            immediate_reward +  # Immediate effect of action
            gamma * (future_reward + next_q) -  # Discounted future effects
            current_q
        )



    
    def _get_valid_actions(self, state, dst) -> List[int]:
        """Get valid next nodes from current state"""
        current_node, _, path_nodes = state
        valid_actions = []
        
        # Get neighbors from topology
        for neighbor in self.topology.neighbors(current_node):
            if neighbor not in path_nodes:  # Avoid loops
                valid_actions.append(neighbor)
                
        return valid_actions
    
    
    def _get_state_key(self, state) -> tuple:
        """Create hashable key for state"""
        node, length, path = state
        return (node, length, tuple(path))
    
    def _add_initial_actions(self, state, dst, queue):
        """Add initial actions to priority queue"""
        valid_actions = self._get_valid_actions(state, dst)
        for action in valid_actions:
            f_cost = self._calculate_f_cost(state, action)
            queue.put((f_cost, (state, action)))



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
        #self.router = QLearningRouter(topology)
        self.qstar_router = QStarRouter(topology, MAX_PATH_LEN)

    def Run(self, rate):
        global topology
        
        #for count in range(1, NUM_OF_REQUESTS + 1):
        for count in tqdm(range(1, NUM_OF_REQUESTS+1), desc="Processing Requests"):
            yield self.env.timeout(self.random.expovariate(rate))
            src, dst = self.random.sample(self.nodes, 2)
            bandwidth = int(self.random.uniform(1, 200))
            holding_time = self.random.expovariate(HOLDING_TIME)
            
            # Use Q-Learning to find routes
            paths = self.qstar_router.find_k_paths(src, dst, N_PATH)
            flag = 0
            #print("paths",paths)
            
            for path in paths:
                distance = int(utils.Distance(topology, path))
                #print("distance",distance)
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
                            #self.update_q_values_for_path(path, OSNR_reward)
                            utils.FirstFit(topology,count, self.check_path[1],self.check_path[2],path)
                            self.connModulationInfo[count] = m
                            spectro = [self.check_path[1], self.check_path[2]]
                            desalocate = Desalocate(self.env)
                            self.env.process(desalocate.Run(count,path,spectro,holding_time))
                            flag=1
                            break
                        #else:
                            #OSNR_reward = -1
                            #self.update_q_values_for_path(path, OSNR_reward)
                    
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
            previous_link_fragmentation = self.qstar_router.global_link_availability[current, next_node]
            previous_link_availability = self.qstar_router.global_link_fragmentation[current,next_node]

            new_link_availability = self.qstar_router.compute_link_availability(current, next_node)
            new_link_fragmentation = self.qstar_router.calculate_link_fragmentation(current, next_node)

            self.qstar_router.global_link_availability[current,next_node] = new_link_availability
            self.qstar_router.global_link_fragmentation[current,next_node] = new_link_fragmentation
            
            # Calculate reward for successful path
            reward = osnr + new_link_availability - new_link_fragmentation
            
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
