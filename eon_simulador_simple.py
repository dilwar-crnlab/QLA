
import simpy
from random import *
from config import *
import numpy as np
import networkx as nx
import math
from itertools import islice

RANDOM_SEED = [50,60,70,80,90,100,110,120,130,140]
MAX_TIME = 10000000  
ERLANG_MIN = 100
ERLANG_MAX = 180
ERLANG_INC = 20
REP = 10
NUM_OF_REQUESTS = 100000
BANDWIDTH = [10,20,40,80,160,200,400]
CLASS_TYPE = [1,2,3]
CLASS_WEIGHT = [0.15, 0.25, 0.60]
TOPOLOGY = 'nsfnet'
HOLDING_TIME = 2.0
SLOTS = 300
SLOT_SIZE = 12.5
N_PATH = 1

topology = nx.read_weighted_edgelist('topology/' + TOPOLOGY, nodetype=int)

class Desalocate(object):
	def __init__(self, env):
		self.env = env
	def Run(self, count, path, spectro, holding_time):
		global topology
		yield self.env.timeout(holding_time)
		for i in range(0, (len(path)-1)):
			for slot in range(spectro[0],spectro[1]+1):
				topology[path[i]][path[i+1]]['capacity'][slot] = 0

class Simulador(object):
	def __init__(self, env):
		self.env = env
		global topology
		for u, v in list(topology.edges):
			topology[u][v]['capacity'] = [0] * SLOTS
		self.nodes = list(topology.nodes())
		self.random = Random()
		self.NumReqBlocked = 0 


	def Run(self, rate):
		global topology
		

		for count in range(1, NUM_OF_REQUESTS + 1):
			yield self.env.timeout(self.random.expovariate(rate))
			src, dst = self.random.sample(self.nodes, 2)
			bandwidth = self.random.choice(BANDWIDTH)
			holding_time = self.random.expovariate(HOLDING_TIME)
			
			paths = self.find_k_path[src,dst]
			flag = 0
			for i in range(N_PATH):
				distance = int(self.Distance(paths[i]))
				num_slots = int(math.ceil(self.Modulation(distance, bandwidth)))
				self.check_path = self.PathIsAble(num_slots,paths[i])
				if self.check_path[0] == True:
					self.cont_req += 1
					self.FirstFit(count, self.check_path[1],self.check_path[2],paths[i])
					spectro = [self.check_path[1], self.check_path[2]]
					desalocate = Desalocate(self.env)
					self.env.process(desalocate.Run(count,paths[i],spectro,holding_time))
					flag = 1
					break 
			if flag == 0:
					self.NumReqBlocked +=1


	# Calcula a distância do caminho de acordo com os pesos das arestas               
	def Distance(self, path):
		global topology 
		soma = 0
		for i in range(0, (len(path)-1)):
			soma += topology[path[i]][path[i+1]]['weight']
		return (soma)

	
	# Calcula o formato de modulação de acordo com a distância do caminho    
	def Modulation(self, dist, demand):
		if dist <= 500:
			return (float(demand) / float(4 * SLOT_SIZE))
		elif 500 < dist <= 1000:
			return (float(demand) / float(3 * SLOT_SIZE))
		elif 1000 < dist <= 2000:
			return (float(demand) / float(2 * SLOT_SIZE)) 
		else:
			return (float(demand) / float(1 * SLOT_SIZE))

	#Realiza a alocação de espectro utilizando First-fit
	def FirstFit(self,count,i,j,path):
		global topology
		inicio = i 
		fim =j
		for i in range(0,len(path)-1):
			for slot in range(inicio,fim):
				#print slot
				topology[path[i]][path[i+1]]['capacity'][slot] = count
			topology[path[i]][path[i+1]]['capacity'][fim] = 'GB'

	# Verifica se o caminho escolhido possui espectro disponível para a demanda requisitada
	def PathIsAble(self, nslots,path):
		global topology
		cont = 0
		t = 0
		for slot in range (0,len(topology[path[0]][path[1]]['capacity'])):
			if topology[path[0]][path[1]]['capacity'][slot] == 0:
				k = 0
				for ind in range(0,len(path)-1):
					if topology[path[ind]][path[ind+1]]['capacity'][slot] == 0:
						k += 1
				if k == len(path)-1:
					cont += 1
					if cont == 1:
						i = slot
					if cont > nslots:
						j = slot
						return [True,i,j]
					if slot == len(topology[path[0]][path[1]]['capacity'])-1:
							return [False,0,0]
				else:
					cont = 0
					if slot == len(topology[path[0]][path[1]]['capacity'])-1:
						return [False,0,0]
			else:
				cont = 0
				if slot == len(topology[path[0]][path[1]]['capacity'])-1:
					return [False,0,0]
