import numpy as np
import pandas as pd
import copy
from heapq import heapify, heappush, heappop
from collections import deque
from typing import Dict, Any, Tuple, List, Optional
from dataclasses import dataclass, field


class Place:
    def __init__(self, name, in_arcs=None, out_arcs=None, properties={}):
        self.name = name
        self.in_arcs = set() if in_arcs is None else in_arcs
        self.out_arcs = set() if out_arcs is None else out_arcs
        self.properties = properties
        
    def __repr__(self):
        return self.name
    
    
class Transition:
    def __init__(self, name, label, in_arcs=None, out_arcs=None, move_type=None, prob=None, weight=None, properties={}, cost_function=None):
        self.name = name
        self.label = label
        self.in_arcs = set() if in_arcs is None else in_arcs 
        self.out_arcs = set() if out_arcs is None else out_arcs
        self.move_type = move_type
        self.prob = prob
        self.cost_function = cost_function
        self.weight = self.__initialize_weight(weight)
        self.properties = properties
        
        
    def __repr__(self):
        return self.name
    
        
    def __initialize_weight(self, weight):
        if self.prob == 0:
            raise ValueError("Probability cannot be zero.")
        
        if weight is not None:
            return weight
        
        if self.cost_function is None:
            return 1e-6 if self.move_type == 'sync' else 1
        
        return self.cost_function(self.prob)
        
    
class Arc:
    def __init__(self, source, target, weight=1, properties={}):
        self.source = source
        self.target = target
        self.weight = weight
        self.properties = properties
        
    def __repr__(self):
        return self.source.name + ' -> ' + self.target.name 
    
    
class Marking:
    def __init__(self, places=None):
        # Convert `places` to a tuple if it's not None, otherwise initialize an empty tuple
        if places is None:
            self.places = tuple()
        else:
            # Ensure `places` is a tuple. If it's not, convert it to a tuple.
            self.places = tuple(places) if not isinstance(places, tuple) else places
    
    def __repr__(self):
        return str(self.places)
    
    
class Node:
    def __init__(self, marking):
        self.marking = marking
        self.neighbors = set()
    
    def __repr__(self):
        return str(self.marking)
    
    def add_neighbor(self, node, transition):
        self.neighbors.add((node, transition)) 
        
        
class Edge:
    def __init__(self, name, source_marking, target_marking, move_type):
        self.name = name
        self.source_marking = source_marking
        self.target_marking = target_marking
        self.move_type = move_type
        
        
    def __repr__(self):
        return f'{self.source_marking} -> {self.name} -> {self.target_marking}'
    
    
class Graph:
    def __init__(self, nodes = None, edges = None, starting_node = None, ending_node = None):
        self.nodes = list() if nodes is None else nodes
        self.edges = list() if edges is None else edges
        self.starting_node = starting_node
        self.ending_node = ending_node
        self.nodes_indices = {}
        
    def __repr__(self):
        return f'Nodes:{self.nodes}, \n edges:{self.edges}'
    
    def __get_markings(self):
        return set([node.marking for node in self.nodes])
    
    def add_node(self, node):
        self.nodes.append(node)
        self.nodes_indices[node.marking] = len(self.nodes) - 1
        
    def add_edge(self, edge): 
        self.edges.append(edge)
        
# Good working solution - without vector reusing
# @dataclass
# class AStarSearchNode:
#     marking: Any = field(compare=False)
#     g_cost: float = field(compare=False)
#     h_cost: float = field(compare=False)
#     f_cost: float = field(init=False, compare=False)
#     path_prefix: Optional[List[Any]] = field(default_factory=list, compare=False)
#     parent: Optional['AStarSearchNode'] = field(default=None, compare=False)
#     transition_to_ancestor: Optional[Any] = field(default=None, compare=False)
#     total_model_moves: int = field(default=0, compare=False)

#     def __post_init__(self):
#         self.f_cost = self.g_cost + self.h_cost

#     def __lt__(self, other):
#         if self.f_cost == other.f_cost:
#             # Tiebreaker: prefer nodes with more model moves
#             return self.total_model_moves > other.total_model_moves
#         return self.f_cost < other.f_cost


class PetriNetNode:
    def __init__(self,  marking_places, parent=None, transition_to_parent=None, depth=np.inf):
        self.marking_places = marking_places
        self.parent = parent 
        self.transition_to_parent = transition_to_parent
        # self.direct_transition_labels = set()  
        self.depth = depth
    
    def __lt__(self, other):
        # Define a logical comparison for nodes, for example by depth
        return self.depth < other.depth  
        

@dataclass
class AStarSearchNode:
    marking: Any = field(compare=False)
    g_cost: float = field(compare=False)
    h_cost: float = field(compare=False)
    f_cost: float = field(init=False, compare=False)
    path_prefix: Optional[List[Any]] = field(default_factory=list, compare=False)
    parent: Optional['AStarSearchNode'] = field(default=None, compare=False)
    transition_to_ancestor: Optional[Any] = field(default=None, compare=False)
    total_model_moves: int = field(default=0, compare=False)
    solution_vector: Optional[np.ndarray] = field(default=None, compare=False)
    have_exact_known_solution: bool = field(default=False, compare=False)

    def __post_init__(self):
        self.f_cost = self.g_cost + self.h_cost

    def __lt__(self, other):
        if self.have_exact_known_solution != other.have_exact_known_solution:
            return self.have_exact_known_solution > other.have_exact_known_solution
        return self.f_cost < other.f_cost


@dataclass
class AStarIncrementalSearchNode:
    marking: Any = field(compare=False)
    g_cost: float = field(compare=False)
    h_cost: float = field(compare=False)
    f_cost: float = field(init=False, compare=False)
    parent: Optional['AStarIncrementalSearchNode'] = field(default=None, compare=False)
    transition_to_ancestor: Optional[Any] = field(default=None, compare=False)
    solution_vector: Optional[np.ndarray] = field(default=None, compare=False)
    has_exact_heuristic: bool = field(default=False, compare=False)
    n_events_explained: int = field(default=0, compare=False)

    def __post_init__(self):
        self.f_cost = self.g_cost + self.h_cost

    def __lt__(self, other):
        if self.f_cost == other.f_cost:
            return self.has_exact_heuristic > other.has_exact_heuristic
        return self.f_cost < other.f_cost

    def update_heuristic(self, new_h_cost, new_solution_vector):
        self.h_cost = new_h_cost
        self.solution_vector = new_solution_vector
        self.has_exact_heuristic = new_solution_vector is not None
        self.f_cost = self.g_cost + self.h_cost


@dataclass(order=True)
class ReachNode:
    f_cost: float = field(init=False)  # Total cost (f = g + h)
    marking: Any = field(compare=False)
    i: int = field(compare=False)
    gamma: List[Any] = field(compare=False)
    c: float = field(compare=False)  # g cost
    h: float = field(compare=False)  # h cost

    def __post_init__(self):
        self.f_cost = self.c + self.h


class search_node_new:
    def __init__(self, marking, dist=np.inf, ancestor=None, transition_to_ancestor=None, path_prefix=None,
                 trace_activities_multiset=None, heuristic_distance=None, total_model_moves=0):
        self.dist = dist
        self.ancestor = ancestor
        self.transition_to_ancestor = transition_to_ancestor
        self.marking = marking
        self.path_prefix = path_prefix if path_prefix is not None else []
        self.trace_activities_multiset = trace_activities_multiset
        # Initialize heuristic_distance with a default of 0 if not provided
        self.heuristic_distance = heuristic_distance if heuristic_distance is not None else 0
        self.total_model_moves = total_model_moves
        
    def __lt__(self, other):
        # First compare based on the sum of dist and heuristic_distance
        if (self.dist + self.heuristic_distance) == (other.dist + other.heuristic_distance):
            # If they are equal, compare based on total_model_moves (larger first)
            return self.total_model_moves > other.total_model_moves
        return (self.dist + self.heuristic_distance) < (other.dist + other.heuristic_distance)


    def __repr__(self):
        return f'Node: {self.marking}, dist: {self.dist}, heuristic: {self.heuristic_distance}'    
    
    
class PetriNet:
    def __init__(self, name='net', places=None, transitions=None, arcs=None, properties={}, conditioned_prob_compute=False):
        self.name = name
        self.transitions = list() if transitions is None else transitions
        self.places = list() if places is None else places
        self.arcs = list() if arcs is None else arcs
        self.properties = properties
        self.init_mark = None
        self.final_mark = None
        self.reachability_graph = None
        self.places_indices = {self.places[i].name:i for i in range(len(self.places))}
        self.transitions_indices = {self.transitions[i].name:i for i in range(len(self.transitions))}
        self.cost_function = None
        self.conditioned_prob_compute = conditioned_prob_compute
        self.mandatory_transitions_map = None
        self.alive_transitions_map = None   
    
    def construct_reachability_graph(self):   
        curr_mark = self.init_mark
        curr_node = Node(curr_mark.places)
        self.reachability_graph = Graph()
        if self.final_mark is not None:
            self.reachability_graph.ending_node = Node(self.final_mark.places)
        self.reachability_graph.add_node(curr_node)
        self.reachability_graph.starting_node = curr_node
        available_transitions = self._find_available_transitions(curr_mark.places)
        nodes_to_explore = deque()
        visited_marks = set()
        
        for transition in available_transitions:
            nodes_to_explore.append((curr_mark, transition, curr_node))
            
        visited_marks.add(curr_mark.places)

        while nodes_to_explore:
            prev_node_triplet = nodes_to_explore.popleft()
            prev_mark, prev_transition, prev_node = prev_node_triplet[0], prev_node_triplet[1], prev_node_triplet[2]
            assert self.__check_transition_prerequesits(prev_transition, prev_mark.places) == True
            curr_mark = self._fire_transition(prev_mark, prev_transition)
            
            if curr_mark.places in visited_marks:
                node_idx = self.reachability_graph.nodes_indices[curr_mark.places]
                curr_node = self.reachability_graph.nodes[node_idx]
            else:
                curr_node = Node(curr_mark.places)
                
            prev_node.add_neighbor(curr_node, prev_transition)
            self.reachability_graph.add_edge(Edge(prev_transition.name, prev_mark, curr_mark, prev_transition.move_type))
            
            if curr_mark.places in visited_marks:
                 continue
            
            else:
                for transition in self._find_available_transitions(curr_mark.places):
                    nodes_to_explore.append((curr_mark, transition, curr_node))
                        
                visited_marks.add(curr_mark.places) 
                self.reachability_graph.add_node(curr_node)


    
    def construct_synchronous_product(self, trace_model, cost_function):
        '''This func assigns all trace transitions move_type=trace and all model transitions move_type=model
        additionaly, all sync transitions will be assigned move_type=sync '''
        
        self.assign_model_transitions_move_type()   
        trace_model.assign_trace_transitions_move_type()
        sync_places = copy.deepcopy(self.places + trace_model.places)
        sync_transitions = copy.deepcopy(self.transitions + trace_model.transitions)
        sync_arcs = copy.deepcopy(self.arcs + trace_model.arcs)
    
        new_sync_transitions = self._generate_all_sync_transitions(trace_model, cost_function)
        sync_prod = PetriNet('sync_prod', sync_places, sync_transitions, sync_arcs)
    
        sync_prod.add_transitions_with_arcs(new_sync_transitions)
        sync_prod.init_mark = Marking(self.init_mark.places + trace_model.init_mark.places)
        sync_prod.final_mark = Marking(self.final_mark.places + trace_model.final_mark.places)
        self.update_sync_product_trans_names(sync_prod)
        print('Wrong function Dude!! -- def construct_synchronous_product')
        return sync_prod
        
        
    def add_places(self, places):
        if isinstance(places, list):
            self.places += places
        
        else:
            self.places.append(places)
        
        self.__update_indices_p_dict(places)
     
    
    def add_transitions(self, transitions):
        if isinstance(transitions, list):
            self.transitions += transitions
        
        else:
            self.transitions.append(transitions)
        
        self.__update_indices_t_dict(transitions)
       
    
    def add_transitions_with_arcs(self, transitions):
        if isinstance(transitions, list):
            self.transitions += transitions
            for transition in transitions:
                self.arcs += list(transition.in_arcs.union(transition.out_arcs))

        else:
            self.transitions.append(transitions) 
            self.arcs += list(transition.in_arcs.union(transition.out_arcs))

        self.__update_indices_t_dict(transitions)
  

    def add_arc_from_to(self, source, target, weight=None):
            if weight is None:
                arc = Arc(source, target)
            else:
                arc = Arc(source, target, weight)
            source.out_arcs.add(arc)
            target.in_arcs.add(arc)
            self.arcs.append(arc)

    
    def _generate_all_sync_transitions(self, trace_model, cost_function):
        sync_transitions = []
        counter = 1

        for trans in self.transitions:
            # trans.label is guaranteed to be unique in the discovered model (from docs)
            if trans.label is not None:
                # Find in the trace model all the transitions with the same label
                same_label_transitions = self.__find_simillar_label_transitions(trace_model, trans.label)

                for trace_trans in same_label_transitions:
                    new_sync_trans = self.__generate_new_trans(trans, trace_trans, counter, cost_function)
                    sync_transitions.append(new_sync_trans)
                    counter += 1
     
        return sync_transitions
    
    
    def __find_simillar_label_transitions(self, trace_model, activity_label):
        '''Returns all the transitions in the trace with a specified activity label'''
        same_label_trans = [transition for transition in trace_model.transitions if transition.label == activity_label]
                                                                                                   
        return same_label_trans
        
           
    def __generate_new_trans(self, trans, trace_trans, counter, cost_function):
#         name = 'sync_transition_' + str(counter)
        name = f'sync_{trace_trans.name}'
        new_sync_transition = Transition(name=name, label=trans.label, move_type='sync', prob=trace_trans.prob, cost_function=cost_function)
        
        input_arcs = trans.in_arcs.union(trace_trans.in_arcs)
        new_input_arcs = []
        for arc in input_arcs:
            new_arc = Arc(arc.source, new_sync_transition, arc.weight)
            new_input_arcs.append(new_arc)
            
        output_arcs = trans.out_arcs.union(trace_trans.out_arcs)
        new_output_arcs = []
        for arc in output_arcs:
            new_arc = Arc(new_sync_transition, arc.target, arc.weight)
            new_output_arcs.append(new_arc)
       
        new_sync_transition.in_arcs = new_sync_transition.in_arcs.union(new_input_arcs)
        new_sync_transition.out_arcs = new_sync_transition.out_arcs.union(new_output_arcs)
       
        return new_sync_transition        

    
    def __update_indices_p_dict(self, places):
        curr_idx = len(self.places_indices)
        if isinstance(places, list):
            for p in places:
                self.places_indices[p.name] = curr_idx
                curr_idx += 1
        else:
            self.places_indices[places.name] = curr_idx
     
    
    def __update_indices_t_dict(self, transitions):
        curr_idx = len(self.transitions_indices)
        if isinstance(transitions, list):
            for t in transitions:
                self.transitions_indices[t.name] = curr_idx
                curr_idx += 1
        else:
            self.transitions_indices[transitions.name] = curr_idx            
     
    
    def _find_available_transitions(self, mark_tuple):
        '''Input: tuple
           Output: list'''
        
        available_transitions = []
        for transition in self.transitions:
            if self.__check_transition_prerequesits(transition, mark_tuple):
                available_transitions.append(transition)
                
        return available_transitions

    
    def __check_transition_prerequesits(self, transition, mark_tuple):
        for arc in transition.in_arcs:
            arc_weight = arc.weight
            source_idx = self.places_indices[arc.source.name]
            if mark_tuple[source_idx] < arc_weight:
                return False
            
        return True
            
    
    def __assign_trace_transitions_move_type(self):
        for trans in self.transitions:
            trans.move_type = 'trace'
            
    
    def assign_trace_transitions_move_type(self):
        return self.__assign_trace_transitions_move_type()   
    
    
    def assign_model_transitions_move_type(self):
        return self.__assign_model_transitions_move_type()
    
    
    def __assign_model_transitions_move_type(self):
        for trans in self.transitions:
                trans.move_type = 'model'
                
        
    def conformance_checking(self, trace_model, hist_prob_dict=None, lamda=0.5):
        sync_prod = self.construct_synchronous_product(trace_model, self.cost_function)      
        return sync_prod._dijkstra_no_rg_construct(hist_prob_dict, lamda=lamda)
    
    
    def __dijkstra(self):
        distance_min_heap = []
        heapify(distance_min_heap)
#         visited_nodes = set()
        search_graph_nodes = [search_node(node) for node in self.reachability_graph.nodes]
        nodes_idx_dict = {search_node.graph_node.marking:idx for idx, search_node in enumerate(search_graph_nodes)}    
        
        source_node_idx = nodes_idx_dict[self.reachability_graph.starting_node.marking]
        source_node = search_graph_nodes[source_node_idx]
        source_node.dist = 0
        
        for node in search_graph_nodes:
            heappush(distance_min_heap, node)
        
        while distance_min_heap:
            min_dist_node = heappop(distance_min_heap)
            need_heapify = False
            
            for neighbor_transition_tuple in min_dist_node.graph_node.neighbors:
                neighbor, transition = neighbor_transition_tuple[0], neighbor_transition_tuple[1]
                alt_distance = min_dist_node.dist + transition.weight
                neighbor_search_idx = nodes_idx_dict[neighbor.marking]
                    
                if alt_distance < search_graph_nodes[neighbor_search_idx].dist:
                    search_graph_nodes[neighbor_search_idx].dist = alt_distance
                    search_graph_nodes[neighbor_search_idx].ancestor = min_dist_node
                    search_graph_nodes[neighbor_search_idx].transition_to_ancestor = transition
                    need_heapify = True
            
            if need_heapify:
                heapify(distance_min_heap)
        
#         print('ending marking is: ', self.reachability_graph.ending_node.marking)
#         print('nodes_idx_dict is: ', nodes_idx_dict)
        final_mark_idx = nodes_idx_dict[self.reachability_graph.ending_node.marking]
        curr_node = search_graph_nodes[final_mark_idx]
        path = []

        while curr_node is not source_node:
#             path.append(curr_node.transition_to_ancestor.label)            
            path.append(curr_node.transition_to_ancestor.name)
            curr_node = curr_node.ancestor
        
#         print(f'Shortest path len: {search_graph_nodes[final_mark_idx].dist}, \n Optimal alignment: {path[::-1]}')
        return path[::-1], search_graph_nodes[final_mark_idx].dist
    
    
    def _dijkstra_no_rg_construct(self, prob_dict, lamda=0.5, return_final_marking=False):
        distance_min_heap = []
        heapify(distance_min_heap) 
        curr_node = search_node_new(self.init_mark, dist=0)
        heappush(distance_min_heap, curr_node)        
        marking_distance_dict = {}
        visited_markings = set()
        
        if prob_dict is None:
            prob_dict = {}
            
        while distance_min_heap:
            min_dist_node = heappop(distance_min_heap)

            if min_dist_node.marking.places in visited_markings:
                continue
                
            if min_dist_node.marking.places == self.final_mark.places:
                break
              
            available_transitions = self._find_available_transitions(min_dist_node.marking.places)
            for transition in available_transitions:
                new_marking = self._fire_transition(min_dist_node.marking, transition)
                
                if new_marking.places in visited_markings:
                    continue
                                        
                if new_marking in visited_markings:
                    continue
                    
                conditioned_transition_weight = self.compute_conditioned_weight(min_dist_node.path_prefix, transition, prob_dict, lamda=lamda)
                if new_marking.places not in marking_distance_dict or marking_distance_dict[new_marking.places] > min_dist_node.dist                                                                                                                                      + conditioned_transition_weight: 
                    new_path_prefix = min_dist_node.path_prefix + transition.label if transition.label is not None else min_dist_node.path_prefix
                    
                    new_node = search_node_new(new_marking,
                                               dist=min_dist_node.dist+conditioned_transition_weight,
                                               ancestor=min_dist_node,
                                               transition_to_ancestor=transition,
                                               path_prefix=new_path_prefix)
                    
                    marking_distance_dict[new_marking.places] = new_node.dist
                    heappush(distance_min_heap, new_node)
            
            visited_markings.add(min_dist_node.marking.places)
            
                
        shortest_path = []    
        curr_node = min_dist_node
        while curr_node.ancestor:
            shortest_path.append(curr_node.transition_to_ancestor.name)  
            curr_node = curr_node.ancestor
        
        if return_final_marking: #TO DO: need to include overlap in the code 
            return shortest_path[::-1], min_dist_node.dist, self.marking.place
        
        return shortest_path[::-1], min_dist_node.dist    
                    


    def _fire_transition(self, mark, transition):
        '''Input: Mark object or tuple, Transition object
        Output: Marking object''' 

        # Check if mark is a tuple or an instance of Marking, and get the places accordingly
        if isinstance(mark, tuple):
            places = mark
        elif isinstance(mark, Marking):  # Assuming Marking is a class you've defined
            places = mark.places
        else:
            raise TypeError("Expected mark to be either a tuple or Marking instance")

        subtract_mark = [0] * len(places)
        for arc in transition.in_arcs:
            place_idx = self.places_indices[arc.source.name]
            subtract_mark[place_idx] -= arc.weight
        
        add_mark = [0] * len(places)
        for arc in transition.out_arcs:
            place_idx = self.places_indices[arc.target.name]
            add_mark[place_idx] += arc.weight
  
        new_mark = tuple([sum(x) for x in zip(places, subtract_mark, add_mark)])
        for elem in new_mark:
            if elem < 0:
                print(f'The original mark was: {mark}, subtracting: {subtract_mark}, adding: {add_mark}, \
resulting in: {new_mark}, during transition: {transition.name}')

        new_mark_obj = Marking(new_mark)
        return new_mark_obj

    def convert_marking_to_pm4py(self, marking: Any) -> Dict[Any, int]:
        return {self.reverse_place_mapping[idx]: tokens 
                for idx, tokens in enumerate(marking.places) 
                if tokens > 0}
    
      
    def compute_conditioned_weight(self, path_prefix, transition, prob_dict, max_length, lamda=0.5):
        if not prob_dict or not path_prefix or transition.label is None:
            return transition.weight
    
        transition_weight = transition.weight
        transition_label = transition.label
        path_prefix_tuple = tuple(path_prefix)
    
        def adjusted_weight(prefix):
            if transition_label in prob_dict[prefix]:
                return (1 - lamda) * (1 - prob_dict[prefix][transition_label]) + lamda * transition_weight
            return (1 - lamda) + lamda * transition_weight
    
        if path_prefix_tuple in prob_dict:
            return adjusted_weight(path_prefix_tuple)
    
        longest_prefix = self.find_longest_prefix(path_prefix_tuple, prob_dict, max_length)
        if longest_prefix:
            return adjusted_weight(longest_prefix)
    
        return 1  # Default cost for a non-sync move
    
    def find_longest_prefix(self, path_prefix, prob_dict, max_length):
        for i in range(min(len(path_prefix), max_length), 0, -1):
            sub_prefix = path_prefix[-i:]
            if sub_prefix in prob_dict:
                return sub_prefix
        return None