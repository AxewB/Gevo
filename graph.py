import math
import warnings
import numpy as np
import networkx as nx
from typing import Iterable
from enum import StrEnum, auto


def ring_graph(num_nodes: int, degree: int) -> nx.Graph:

    if degree >= num_nodes:
        raise ValueError("degree >= num_nodes, choose smaller k or larger n")
    if degree % 2:
        warnings.warn(
            f"Odd k in ring_graph(). Using degree = {degree - 1} instead.",
            category=RuntimeWarning,
            stacklevel=2
        )

    g = nx.Graph()
    nodes = list(range(num_nodes))  # nodes are labeled 0 to n-1

    for j in range(1, degree // 2 + 1):  # connect each node to k/2 neighbors
        targets = nodes[j:] + nodes[0:j]  # first j nodes are now last
        g.add_edges_from(zip(nodes, targets))

    return g


''' Author: AksenovIV '''
def create_possible_edges(size: int | list, num_nodes: int = None, use_neighbours: bool = False) -> list: 
    """Функция для создания списка возможных ребер

    Параметры:
        size (int | list): Либо количество вершин, либо список вершин
        num_nodes (int, optional): Общее число вершин в графе. Требуется, если size - список.
        use_neighbours (bool, optional): Учитывать ли соседей. По стандарту False.

    Возвращает:
        list: Список из ребер типа tuple(int, int)
    """
    if (type(size) == int):
        return [
            (v1, v2)
            for v1 in list(range(size))
            for v2 in list(range(v1, size))
            if (v1 != v2) and ((v1 + 1) % size != v2) and (v1 != (v2 + 1) % size)
        ]
    elif ((type(size)) == list):
        size = list(map(lambda x: x % num_nodes, size))
        size.sort()
        return [
            (v1, v2)
            for i, v1 in enumerate(size)
            for v2 in size[i + 1:]
            if (v1 != v2) and (
                use_neighbours != False or
                ((v1 + 1) % num_nodes != v2) and (v1 != (v2 + 1) % num_nodes)
            )
        ]
    else:
        raise("Wrong data type while creating possible edges")

''' Author: AksenovIV '''
def add_edge_to_graph(g: nx.Graph, edges: list) -> bool:
    """Добавление ребра в граф, если такого ребра не существует

    Параметры:
        g (nx.Graph): граф, в который будет добавляться ребро
        edges (list): список ребер (желательно перемешанный для обеспечения случайности), 
                      из которых будет выбираться ребро для добавления в граф

    Возвращает:
        bool: возвращает результат добавления: True - если ребро добавлено, False - если ребро уже существует
    """
    edge = edges.pop()
    if (g.has_edge(*edge)):
        return False
    else: 
        g.add_edge(*edge)
        return True
            

''' Author: AksenovIV '''
def ring_graph_with_subgraph(num_nodes: int, 
                             additional_edges: int, 
                             subgraph_edge_fraction: float, 
                             random_nodes_position: bool,
                             num_nodes_subgraph: int = None) -> tuple[nx.Graph, int]:
    """Функция для создания графа с плотным подграфом

    Параметры:
        num_nodes (int): Количество вершин в графе
        additional_edges (int): Количество дополнительных ребер
        subgraph_edge_fraction (float): Доля ребер в подграфе
        random_nodes_position (bool): Режим случайного выбора вершин
        num_nodes_subgraph (int, optional): Количество вершин в подграфе. По стандарту None.

    Возвращает:
        tuple[nx.Graph, int]: Результирующий граф и список вершин подграфа
    """
    g = ring_graph(num_nodes, 2)
    
    # Если количество вершин для подграфа не передано - делаем стандартное значение, равное половине всех вершин
    if (num_nodes_subgraph == None):
        num_nodes_subgraph = num_nodes // 2
        
    possible_edges = create_possible_edges(num_nodes)
    subgraph_nodes = list()
    possible_edges_subgraph = list()

    # random_nodes_position = True: Режим случайного выбора вершин
    # Режим случайного выбора первой вершины, остальные идут подряд
    if (random_nodes_position):
        subgraph_nodes = list(range(num_nodes))
        np.random.shuffle(subgraph_nodes)
        subgraph_nodes = subgraph_nodes[:num_nodes_subgraph]
        subgraph_nodes.sort()
        possible_edges_subgraph = create_possible_edges(subgraph_nodes, num_nodes)
    else:
        subgraph_start_node = np.random.randint(0, num_nodes)
        subgraph_nodes = list(range(subgraph_start_node, subgraph_start_node + num_nodes_subgraph))
        subgraph_nodes = list(map(lambda x: x % num_nodes, subgraph_nodes))
        possible_edges_subgraph = create_possible_edges(subgraph_nodes, num_nodes)
        
    
    # Перемешиваем сочетания, чтобы был элемент случайности
    np.random.shuffle(possible_edges)
    np.random.shuffle(possible_edges_subgraph)
    
    for j in range(0, additional_edges):
        if (np.random.random() < subgraph_edge_fraction and len(possible_edges_subgraph) > 0):
            while(len(possible_edges_subgraph) > 0):
                if (add_edge_to_graph(g, possible_edges_subgraph)):
                    break
        else: 
            while(len(possible_edges) > 0):
                if (add_edge_to_graph(g, possible_edges)):
                    break

    return g, subgraph_nodes

''' Author: AksenovIV '''
def connect_in_ring(graph: nx.Graph, nodes: list) -> None:
    """Функция для соединения списка ребер в графе в кольцо

    Параметры:
        graph (nx.Graph): граф, в котором будут соединяться вершины
        nodes (list): список вершин графа, которые будут соединены в кольцо
    """
    for j in range(1, 2):
        targets = nodes[1:] + nodes[0:1] 
        graph.add_edges_from(zip(nodes, targets))

''' Author: AksenovIV '''
def two_city(num_nodes: int, 
             additional_edges: int, 
             subgraph_edge_fraction: float, 
             num_nodes_subgraph: int, 
             num_shared_edges: int) -> nx.Graph:
    """Функция для создания двухгородного графа

    Параметры:
        num_nodes (int): Количество вершин в графе
        additional_edges (int): Количество дополнительных ребер
        subgraph_edge_fraction (float): Плотность подграфа
        num_shared_edges (int): Количество общих ребер
        num_nodes_subgraph (int): Количество вершин в подграфе

    Возвращает:
        nx.Graph: Результирующий граф из двух городов
    """

    # Убираем общие ребра, чтобы количество дополнительных ребер было одинаковым
    additional_edges -= num_shared_edges 
    
    # Если количество вершин для подграфа не передано - делаем стандартное значение, равное половине всех вершин
    if (num_nodes_subgraph == None):
        num_nodes_subgraph = num_nodes // 2
    g = nx.Graph()
    
    # Создаем два подграфа
    all_nodes = list(range(num_nodes))
    g_nodes = [
        all_nodes[:num_nodes_subgraph], 
        all_nodes[num_nodes_subgraph:]
    ]
    # Возможные ребра только в подграфах
    g_edges = [ 
        create_possible_edges(g_nodes[j], num_nodes) for j in range(len(g_nodes)) 
    ] 
    # Возможные ребра между подграфами
    shared_edges = [
        (v1, v2) 
        for v1 in g_nodes[0] 
        for v2 in g_nodes[1]
    ]
    
    # Перемешиваем ребра для добавления случайности их расположения
    np.random.shuffle(g_edges[0])
    np.random.shuffle(g_edges[1])
    np.random.shuffle(shared_edges)
    
    # Соединяем вершины в подграфах в кольцо с целью обеспечения связности
    connect_in_ring(g, g_nodes[0])
    connect_in_ring(g, g_nodes[1])

    # Распределяем дополнительные ребра между подграфами
    for _ in range(additional_edges):
        if (np.random.random() < subgraph_edge_fraction and len(g_edges[0]) > 0):
            while(len(g_edges[0]) > 0):
                if (add_edge_to_graph(g, g_edges[0])):
                    break
        else:
            while(len(g_edges[1]) > 0):
                if (add_edge_to_graph(g, g_edges[1])):
                    break

    # Добавляем ребра между двумя подграфами
    for _ in range(num_shared_edges):
        add_edge_to_graph(g, shared_edges)

    return g

class GraphType(StrEnum):
    complete = auto()
    ring = auto()
    roc = auto()
    er = auto()
    ring_subgraph = auto()
    two_city = auto()


class Graph:
    graph = nx.empty_graph()
    num_nodes: int
    adj_matrix: np.ndarray
    name: str
    graph_type: GraphType

    layout: dict ### Способ отображения графа
    
    @staticmethod
    def _get_adjacency_matrix(graph):
        return nx.to_numpy_array(graph).astype(np.float32)

    def is_connected_after_node_removal(self, nodes_to_remove: Iterable) -> bool:
        g = self.graph.copy()
        g.remove_nodes_from(nodes_to_remove)
        return nx.is_connected(g)

    def __repr__(self):
        return self.name

    def init_layout(self):
        ''' Стандартный способ отображения графа '''
        self.layout = nx.kamada_kawai_layout(self.graph)


class CompleteGraph(Graph):
    graph_type = GraphType.complete

    def __init__(self, num_nodes: int):
        self.num_nodes = num_nodes
        self.graph = nx.complete_graph(self.num_nodes)
        self.name = f"{self.graph_type}(n={self.num_nodes})"
        self.adj_matrix = self._get_adjacency_matrix(self.graph)
        self.init_layout()

    

class RingGraph(Graph):
    graph_type = GraphType.ring

    def __init__(self, num_nodes: int, degree: int):
        self.num_nodes = num_nodes
        self.degree = degree
        self.graph = ring_graph(self.num_nodes, self.degree)
        self.name = f"{self.graph_type}(n={self.num_nodes},k={self.degree})"
        self.adj_matrix = self._get_adjacency_matrix(self.graph)
        self.init_layout()

    def init_layout(self):
        self.layout = nx.circular_layout(self.graph)

''' Author: Aksenov IV
Кольцевой граф с плотным подграфом
'''
class RingGraphWithSubgraph(Graph):
    graph_type = GraphType.ring_subgraph
    subgraph_nodes: list
    
    def __init__(self, num_nodes: int, additional_edges: int, subgraph_edge_fraction: float, random_nodes_position: bool, num_nodes_subgraph: int = None):
        self.num_nodes = num_nodes
        self.graph, self.subgraph_nodes = ring_graph_with_subgraph(self.num_nodes, 
                                                                   additional_edges, 
                                                                   subgraph_edge_fraction, 
                                                                   random_nodes_position, 
                                                                   num_nodes_subgraph)
        self.name = f"{self.graph_type}(n={self.num_nodes},g={len(self.subgraph_nodes)},p={subgraph_edge_fraction},pos={'random' if random_nodes_position else 'row'})"
        self.adj_matrix = self._get_adjacency_matrix(self.graph)
        self.init_layout()

    def init_layout(self):
        external_nodes = [ node for node in self.graph.nodes if (node not in self.subgraph_nodes) ]
        external_layout = nx.circular_layout(external_nodes, scale=1)
        internal_layout = nx.circular_layout(self.subgraph_nodes, scale=0.8)
        self.layout = (external_layout | internal_layout)

''' Author: Aksenov IV
Граф двух городов
'''
class TwoCityGraph(Graph):
    graph_type = GraphType.two_city
    num_nodes_subgraph: int

    def __init__(self, num_nodes: int, additional_edges: int, subgraph_edge_fraction: float, num_nodes_subgraph: int = None, num_shared_edges: int = 1):
        self.num_nodes = num_nodes
        self.graph = two_city(self.num_nodes, additional_edges, subgraph_edge_fraction, num_nodes_subgraph, num_shared_edges)
        self.num_nodes_subgraph = num_nodes_subgraph if num_nodes_subgraph is not None else num_nodes // 2
        self.name = f"{self.graph_type}(n={self.num_nodes},a_e={additional_edges},sh_e={num_shared_edges},p={subgraph_edge_fraction})"
        self.adj_matrix = self._get_adjacency_matrix(self.graph)
        self.init_layout()

    def init_layout(self):
        left = nx.kamada_kawai_layout(list(range(0, self.num_nodes_subgraph)), center=(0, 0))
        right = nx.kamada_kawai_layout(list(range(self.num_nodes_subgraph, self.graph.number_of_nodes())),center=(2.5, 0))

        self.layout = (left | right)

class RocGraph(Graph):
    graph_type = GraphType.roc

    def __init__(self, num_cliques: int, clique_size: int):
        self.num_nodes = num_cliques * clique_size
        self.num_cliques = num_cliques
        self.clique_size = clique_size
        self.graph = nx.ring_of_cliques(self.num_cliques, self.clique_size)
        self.name = f"{self.graph_type}(k={self.num_cliques},l={self.clique_size})"
        self.adj_matrix = self._get_adjacency_matrix(self.graph)
        self.init_layout()


class ErGraph(Graph):
    graph_type = GraphType.er

    def __init__(self, num_nodes: int, probability: float = None, num_edges: int = None):
        self.num_nodes = num_nodes
        if (probability is None) == (num_edges is None):
            raise ValueError("Supply either p or m")
        if probability is not None:
            self.probability = probability
            self.graph = nx.erdos_renyi_graph(self.num_nodes, self.probability)
            while not nx.is_connected(self.graph):
                self.graph = nx.erdos_renyi_graph(self.num_nodes, self.probability)
            self.name = f"{self.graph_type}(n={self.num_nodes},p={self.probability})"
        if num_edges is not None:
            self.num_edges = num_edges
            self.graph = nx.gnm_random_graph(self.num_nodes, self.num_edges)
            while not nx.is_connected(self.graph):
                self.graph = nx.gnm_random_graph(self.num_nodes, self.num_edges)
            self.name = f"{self.graph_type}(n={self.num_nodes},m={self.num_edges})"
        self.adj_matrix = self._get_adjacency_matrix(self.graph)
        self.init_layout()
