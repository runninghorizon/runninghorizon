from Classes import *
from utils import *
from preprocessing import *
import random
import time
from collections import OrderedDict
from collections import Counter
from collections import deque
from random import sample
import math
from random import seed
import scipy
from scipy.optimize import linprog
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpStatus, PULP_CBC_CMD
from typing import Any, List, Tuple, Set, Optional

try:
    import gurobipy as gp
    from gurobipy import GRB
    GUROBI_AVAILABLE = True
except ImportError:
    GUROBI_AVAILABLE = False
    
def split_df(df, max_len=10, discovery_ratio=0.1):
    # Filter out groups that are longer than max_len
    df = df.groupby('case:concept:name').filter(lambda x: len(x) <= max_len)

    # Get a list of unique 'case:concept:name' values
    cases = df['case:concept:name'].unique()

    # Split the cases into train and test sets
    train_cases, test_cases = train_test_split(cases, test_size=discovery_ratio, random_state=42)

    # Create train and test DataFrames
    train_df = df[df['case:concept:name'].isin(train_cases)]
    test_df = df[df['case:concept:name'].isin(test_cases)]

    return train_df, test_df



def discover_process_model(df_train, return_markings=True):
        df = prepare_df_cols_for_discovery(df_train)
        net, init_marking, final_marking = pm4py.discover_petri_net_inductive(df)
        
        if return_markings:
            return net, init_marking, final_marking
        return net

      
    
def discover_process_tree(df_train, return_tree=False, viz_tree=True):
    df = prepare_df_cols_for_discovery(df_train)
    tree = pm4py.discover_process_tree_inductive(df)
    
    if viz_tree:
        pm4py.view_process_tree(tree)
        
    if return_tree:
        return tree
    
    
def prepare_determ_df_for_conformance(df):
    df_copy = df.copy()
    df_copy['probs'] = [[1.0] for _ in range(len(df_copy))]
    
    return df_copy


def split_dataframe(df, window_len, overlap=0):
    if overlap >= window_len:
        raise ValueError("Overlap must be smaller than window length")
    
    step_size = window_len - overlap
    return [df[i:i + window_len] for i in range(0, df.shape[0], step_size)]
 
    
class MarkingAncestorNode():
    def __init__(self, ancestor_marking_node, dist, alignment):
        self.ancestor_marking = ancestor_marking_node
        self.dist = dist
        self.alignment = alignment if alignment is not None else []

        
class SyncProduct(PetriNet):
   
    def __init__(self, net, trace_model, cost_function=None, init_mark=None,
                 final_mark=None, fancy_cost_function=None):
           
        super().__init__()  
        net.assign_model_transitions_move_type()
        trace_model.assign_trace_transitions_move_type()
        
        self.net = net
        self.trace_model = trace_model
        self.places = self.net.places + self.trace_model.places
        self.transitions = self.net.transitions + self.trace_model.transitions
        self.arcs = self.net.arcs + self.trace_model.arcs
        
        new_sync_transitions = self.net._generate_all_sync_transitions(self.trace_model, cost_function)
        self.add_transitions_with_arcs(new_sync_transitions)
        self.update_sync_product_trans_names()
        
        self.places_indices = {place.name: idx for idx, place in enumerate(self.places)}
        self.transitions_indices = {transition.name: idx for idx, transition in enumerate(self.transitions)}
        self._incidence_matrix = None  # Initialize as None
        self._consumption_matrix = None
        
        if GUROBI_AVAILABLE:
            self.gp = gp
            self.GRB = gp.GRB
        
        if fancy_cost_function is not None and cost_function is not None:
            raise ValueError("Both cost function and fancy cost function are not None")
            
        if fancy_cost_function is not None:
            for t in self.transitions:
                t.weight = fancy_cost_function(t)        
    
        self.init_mark = Marking(self.net.init_mark.places + self.trace_model.init_mark.places) if init_mark is None else init_mark
        self.final_mark = Marking(self.net.final_mark.places + self.trace_model.final_mark.places) if final_mark is None else final_mark


    def update_sync_product_trans_names(self, sync_product=None):
        if sync_product is None:
            transitions = self.transitions
        else:
            transitions = sync_product.transitions

        for trans in transitions:
            if trans.move_type == 'model':
                if not hasattr(self.net, '_sync_names_updated') or not self.net._sync_names_updated:
                    trans.name = f'(>>, {trans.name})'
            elif trans.move_type == 'trace':
                trans.name = f'({trans.name}, >>)'
            else:
                trans.name = f'({trans.name}, {trans.name})'

        if not hasattr(self.net, '_sync_names_updated'):
            self.net._sync_names_updated = True

        transitions_indices = {transitions[i].name: i for i in range(len(transitions))}

        if sync_product is not None:
            sync_product.transitions_indices = transitions_indices
            return sync_product
        else:
            self.transitions_indices = transitions_indices

    
    def estimate_alignment_heuristic(self, new_marking, trace_activities_multiset=None, tight_breaking_factor=1.00001):
        """
        Estimates the heuristic alignment cost based on the remaining activities in trace_activities_multiset 
        and missing mandatory transitions associated with new_marking.
    
        Args:
            new_marking (Marking): A Marking object representing the current state.
            trace_activities_multiset (Counter): A Counter object representing the frequency of activities in the trace.
            tight_breaking_factor (float): A factor to prioritize synchronous moves over nonsynchronous moves.
    
        Returns:
            float: The estimated heuristic cost based on remaining activity counts and missing mandatory transitions.
        """
        
        # Get the set of mandatory model moves that should occur
        mandatory_model_moves = self.get_mandatory_transitions(new_marking)
        
        if trace_activities_multiset is None:
            # If trace_activities_multiset is None, consider only the missing required transitions
            missing_required_transitions = mandatory_model_moves
            return len(missing_required_transitions) * tight_breaking_factor
        
        net_places = new_marking.places[:len(self.net.places)]
        
        if net_places not in self.net.alive_transitions_map:
            raise KeyError(f"Marking place '{net_places}' not found in net's alive_transitions_map dictionary.")
        
        reachable_transitions = self.net.alive_transitions_map[net_places]['reachable_transitions']
        temp_multiset = trace_activities_multiset.copy()
        
        # Remove reachable transitions from the multiset of trace activities
        for t_label in list(temp_multiset.keys()):
            if t_label in reachable_transitions:
                del temp_multiset[t_label]
        
        heuristic_log_moves_cost = sum(temp_multiset.values())
        
        # Identify missing required transitions by checking those not in the remaining trace
        missing_required_transitions = mandatory_model_moves - set(trace_activities_multiset.keys())
        
        return heuristic_log_moves_cost + len(missing_required_transitions) * tight_breaking_factor
        
    
    def construct_synchronous_product(self, trace_model, cost_function, net_init_mark=None, net_final_mark=None):
        return SyncProduct(net=self, trace_model=trace_model, cost_function=cost_function,
                           net_init_mark=net_init_mark, net_final_mark=net_final_mark)

        
    def conformance_checking(self, trace_model, hist_prob_dict=None, lamda=0.5,
                             partial_conformance=False, return_net_final_marking=False):
        
        sync_prod = self.construct_synchronous_product(trace_model, self.cost_function)

        return sync_prod._dijkstra_no_rg_construct(prob_dict=hist_prob_dict, lamda=lamda,
                                                   partial_conformance=partial_conformance,
                                                   return_net_final_marking=return_net_final_marking)        


    # Original working code 01/08/24
    # def _dijkstra_no_rg_construct(self, prob_dict=None, lamda=0.5, partial_conformance=False,
    #                               return_net_final_marking=False, n_unique_final_markings=1,
    #                               explor_reward=0.001, overlap_size=0, trace_activities_multiset=None,
    #                               net_details_dict=None, use_heuristic_distance=False, trace_recovery=False,
    #                               max_hist_len=None):

    #         distance_min_heap = []
    #         heapify(distance_min_heap) 
            
    #         if use_heuristic_distance:
    #             init_heuristic = self.compute_heuristic_distance(self.init_mark, trace_activities_multiset, 
    #                                                              net_details_dict)

    #         else:
    #             init_heuristic = 0
    #             trace_activities_multiset = set()
                
    #         curr_node = search_node_new(self.init_mark, dist=0,
    #                                     trace_activities_multiset=trace_activities_multiset.copy(),
    #                                     heuristic_distance=init_heuristic, total_model_moves=0)

    #         heappush(distance_min_heap, curr_node)        
    #         marking_distance_dict = {}
    #         visited_markings = set()
    #         final_nodes = []
    #         final_markings_unique = set()
    #         edge_usage_count = {}
    #         nodes_opened = 0
            
    #         while distance_min_heap:
    #             min_dist_node = heappop(distance_min_heap)
                
    #             if min_dist_node.marking.places in visited_markings:
    #                 continue

    #             if partial_conformance:
    #                 if min_dist_node.marking.places[-len(self.trace_model.places):] == self.trace_model.final_mark.places:
    #                     if min_dist_node.marking.places[:len(self.net.places)] not in final_markings_unique:
    #                         final_nodes.append(min_dist_node)
    #                         final_markings_unique.add(min_dist_node.marking.places[:len(self.net.places)])

    #                     if len(final_markings_unique) == n_unique_final_markings: 
    #                         break
    #                     else:
    #                         continue

    #             elif min_dist_node.marking.places == self.final_mark.places:
    #                     final_nodes.append(min_dist_node)
    #                     if len(final_nodes) == n_unique_final_markings: 
    #                         break
    #                     else:
    #                         continue

    #             nodes_opened += 1
    #             available_transitions = self._find_available_transitions(min_dist_node.marking.places)
    #             # print(f'Here are the available transitions for the search node: {[t.name for t in available_transitions]}')
    #             for transition in available_transitions:
    #                 new_marking = self._fire_transition(min_dist_node.marking, transition)

    #                 if new_marking.places in visited_markings:
    #                     continue                              

    #                 conditioned_transition_weight = self.compute_conditioned_weight(min_dist_node.path_prefix,
    #                                                                                 transition,
    #                                                                                 prob_dict,
    #                                                                                 max_length=max_hist_len,
    #                                                                                 lamda=lamda)



    #                 if (new_marking.places not in marking_distance_dict or
    #                     marking_distance_dict[new_marking.places] > min_dist_node.dist + conditioned_transition_weight): 
    #                     new_path_prefix = min_dist_node.path_prefix + [transition.label] if transition.label is not None else min_dist_node.path_prefix

                        
    #                     if use_heuristic_distance:
    #                         if transition.move_type in {'trace', 'sync'}:
    #                             leftover_trace_activities_multiset = min_dist_node.trace_activities_multiset.copy()
    #                             leftover_trace_activities_multiset = subtract_activities(leftover_trace_activities_multiset,
    #                                                                                      transition.label)

    #                             heuristic_distance = self.compute_heuristic_distance(new_marking,
    #                                                                             leftover_trace_activities_multiset,
    #                                                                                  net_details_dict)

    #                         else:
    #                             heuristic_distance = self.compute_heuristic_distance(new_marking,
    #                                                                                  min_dist_node.trace_activities_multiset,
    #                                                                                  net_details_dict)

    #                             leftover_trace_activities_multiset = min_dist_node.trace_activities_multiset
                                
    #                     else:
    #                         heuristic_distance = 0
    #                         leftover_trace_activities_multiset = set()
                            
    #                     if transition.move_type in {'model', 'sync'}:
    #                         total_model_moves = min_dist_node.total_model_moves + 1
    #                     else:
    #                         total_model_moves = min_dist_node.total_model_moves
                            
    #                     new_node = search_node_new(new_marking,
    #                                                dist=min_dist_node.dist + conditioned_transition_weight,
    #                                                ancestor=min_dist_node,
    #                                                transition_to_ancestor=transition,
    #                                                heuristic_distance=heuristic_distance,
    #                                                path_prefix=new_path_prefix,
    #                                                trace_activities_multiset=leftover_trace_activities_multiset,
    #                                                total_model_moves=total_model_moves)

    #                     marking_distance_dict[new_marking.places] = new_node.dist
    #                     heappush(distance_min_heap, new_node)

    #             visited_markings.add(min_dist_node.marking.places)

    #         final_non_overlap_trace_marking = [1 if i == len(self.trace_model.places) - overlap_size - 1 else 0 for i in range(len(self.trace_model.places))]

    #         final_path_dist_marking_triplets = []

    #         for node in final_nodes:
    #             shortest_path = []    
    #             curr_node = node

    #             while partial_conformance and list(curr_node.marking.places[-len(self.trace_model.places):]) != final_non_overlap_trace_marking:
    #                 curr_node = curr_node.ancestor

    #             if trace_recovery:
    #                 alignment_lst = []
    #                 while curr_node.ancestor:
    #                     alignment_lst.append(curr_node.transition_to_ancestor)
    #                     curr_node = curr_node.ancestor
    #                 return alignment_lst[::-1]
                
    #             total_distance = 0
    #             while curr_node.ancestor:
    #                 shortest_path.append(curr_node.transition_to_ancestor)
    #                 total_distance += curr_node.transition_to_ancestor.weight
    #                 curr_node = curr_node.ancestor

    #             if return_net_final_marking:
    #                 final_path_dist_marking_triplets.append((shortest_path[::-1], node.dist,
    #                                                          Marking(node.marking.places[:len(self.net.places)]),
    #                                                          node.heuristic_distance))

    #             else:
    #                 final_path_dist_marking_triplets.append((shortest_path[::-1], node.dist, node.heuristic_distance))   
       
    #         return final_path_dist_marking_triplets, nodes_opened    
    

    def _dijkstra_no_rg_construct(self, prob_dict=None, lamda=0.5, partial_conformance=False,
                                  return_net_final_marking=False, n_unique_final_markings=1,
                                  explor_reward=0.001, overlap_size=0, trace_activities_multiset=None,
                                  use_heuristic_distance=False, trace_recovery=False,
                                  max_hist_len=None):
        distance_min_heap = []
        marking_distance_dict = {}
        visited_markings = set()
        final_nodes = []
        final_markings_unique = set()
        nodes_opened = 0
    
        init_node = self._initialize_dijkstra_node(trace_activities_multiset, use_heuristic_distance)
        heappush(distance_min_heap, init_node)
    
        while distance_min_heap:
            current_node = heappop(distance_min_heap)
            
            if current_node.marking.places in visited_markings:
                continue
    
            if self._is_dijkstra_final_node(current_node, partial_conformance, n_unique_final_markings, final_markings_unique):
                final_nodes.append(current_node)
                if len(final_nodes) == n_unique_final_markings:
                    break
                continue
    
            nodes_opened += 1
            available_transitions = self._find_available_transitions(current_node.marking.places)
    
            for transition in available_transitions:
                new_node = self._create_dijkstra_successor_node(current_node, transition, prob_dict, lamda, max_hist_len,
                                                                use_heuristic_distance)
    
                if self._should_add_dijkstra_node(new_node, marking_distance_dict):
                    marking_distance_dict[new_node.marking.places] = new_node.dist
                    heappush(distance_min_heap, new_node)
    
            visited_markings.add(current_node.marking.places)
    
        return self._process_dijkstra_final_nodes(final_nodes, partial_conformance, overlap_size, trace_recovery, return_net_final_marking), nodes_opened
    
    def _initialize_dijkstra_node(self, trace_activities_multiset, use_heuristic_distance):
        init_heuristic = self.estimate_alignment_heuristic(self.init_mark, trace_activities_multiset) if use_heuristic_distance else 0
        trace_activities_multiset = trace_activities_multiset or set()
        return search_node_new(self.init_mark, dist=0, trace_activities_multiset=trace_activities_multiset.copy(),
                               heuristic_distance=init_heuristic, total_model_moves=0)
    
    def _is_dijkstra_final_node(self, node, partial_conformance, n_unique_final_markings, final_markings_unique):
        if partial_conformance:
            if node.marking.places[-len(self.trace_model.places):] == self.trace_model.final_mark.places:
                model_marking = node.marking.places[:len(self.net.places)]
                if model_marking not in final_markings_unique:
                    final_markings_unique.add(model_marking)
                    return True
        elif node.marking.places == self.final_mark.places:
            return True
        return False
    
    def _create_dijkstra_successor_node(self, current_node, transition, prob_dict, lamda, max_hist_len, use_heuristic_distance):
        new_marking = self._fire_transition(current_node.marking, transition)
        conditioned_transition_weight = self.compute_conditioned_weight(current_node.path_prefix, transition, prob_dict, max_length=max_hist_len, lamda=lamda)
        new_path_prefix = current_node.path_prefix + [transition.label] if transition.label is not None else current_node.path_prefix
    
        heuristic_distance, leftover_trace_activities_multiset = self._compute_dijkstra_heuristic(current_node, transition, new_marking, use_heuristic_distance)
        
        total_model_moves = current_node.total_model_moves + (1 if transition.move_type in {'model', 'sync'} else 0)
    
        return search_node_new(new_marking, dist=current_node.dist + conditioned_transition_weight,
                               ancestor=current_node, transition_to_ancestor=transition,
                               heuristic_distance=heuristic_distance, path_prefix=new_path_prefix,
                               trace_activities_multiset=leftover_trace_activities_multiset,
                               total_model_moves=total_model_moves)

    def _compute_dijkstra_heuristic(self, current_node, transition, new_marking, use_heuristic_distance):
        
        if not use_heuristic_distance:
            return 0, set()
    
        leftover_trace_activities_multiset = current_node.trace_activities_multiset.copy()
        if transition.move_type in {'trace', 'sync'}:
            leftover_trace_activities_multiset = subtract_activities(leftover_trace_activities_multiset, transition.label)
        
        heuristic_distance = self.estimate_alignment_heuristic(new_marking, leftover_trace_activities_multiset)

        return heuristic_distance, leftover_trace_activities_multiset
    
    def _should_add_dijkstra_node(self, new_node, marking_distance_dict):
        return (new_node.marking.places not in marking_distance_dict or
                marking_distance_dict[new_node.marking.places] > new_node.dist)
    
    def _process_dijkstra_final_nodes(self, final_nodes, partial_conformance, overlap_size, trace_recovery, return_net_final_marking):
        final_non_overlap_trace_marking = [1 if i == len(self.trace_model.places) - overlap_size - 1 else 0 for i in range(len(self.trace_model.places))]
        final_path_dist_marking_triplets = []
    
        for node in final_nodes:
            if trace_recovery:
                return self._recover_dijkstra_trace(node)
            
            path, total_distance = self._build_dijkstra_path(node, partial_conformance, final_non_overlap_trace_marking)
            
            if return_net_final_marking:
                final_path_dist_marking_triplets.append((path, node.dist, Marking(node.marking.places[:len(self.net.places)]), node.heuristic_distance))
            else:
                final_path_dist_marking_triplets.append((path, node.dist, node.heuristic_distance))
    
        return final_path_dist_marking_triplets
    
    def _recover_dijkstra_trace(self, node):
        alignment_lst = []
        while node.ancestor:
            alignment_lst.append(node.transition_to_ancestor)
            node = node.ancestor
        return alignment_lst[::-1]
    
    def _build_dijkstra_path(self, node, partial_conformance, final_non_overlap_trace_marking):
        path = []
        total_distance = 0
        curr_node = node
    
        while partial_conformance and list(curr_node.marking.places[-len(self.trace_model.places):]) != final_non_overlap_trace_marking:
            curr_node = curr_node.ancestor
    
        while curr_node.ancestor:
            path.append(curr_node.transition_to_ancestor)
            total_distance += curr_node.transition_to_ancestor.weight
            curr_node = curr_node.ancestor
    
        return path[::-1], total_distance
    
    def astar_search(self, trace_recovery: bool = False) -> Tuple[List[Any], float]:
        if self._incidence_matrix is None:
            self._incidence_matrix = self._get_incidence_matrix()
    
        open_set = []
        heapify(open_set)
        closed_set = {}
        reached_set = {}

        nodes_popped = 0  

        initial_h_cost, initial_solution_vector = self._compute_marking_equation(self.init_mark)
        initial_node = AStarSearchNode(
            marking=self.init_mark,
            g_cost=0,
            h_cost=initial_h_cost,
            solution_vector=initial_solution_vector,
            have_exact_known_solution=True
        )
        heappush(open_set, initial_node)
    
        while open_set:
            current_node = heappop(open_set)
            nodes_popped += 1
            
            if current_node.marking.places == self.final_mark.places:
                return self._process_final_node(current_node, nodes_popped, trace_recovery)
    
            if current_node.marking.places in closed_set:
                continue
    
            closed_set[current_node.marking.places] = current_node.g_cost
    
            for transition in self._find_available_transitions(current_node.marking.places):
                new_marking = self._fire_transition(current_node.marking, transition)
    
                transition_cost = transition.weight
    
                new_g_cost = current_node.g_cost + transition_cost
    
                if (new_marking.places in closed_set or 
                   (new_marking.places in reached_set and reached_set[new_marking.places] <= new_g_cost)):
                   continue
    
                if (current_node.solution_vector is not None and
                    current_node.solution_vector[self.transitions_indices[transition.name]] >= 1):
                    # Reuse solution
                    new_h_cost = current_node.h_cost - transition.weight
                    new_solution_vector = current_node.solution_vector.copy()
                    new_solution_vector[self.transitions_indices[transition.name]] -= 1
                    have_exact_known_solution = True
                else:
                    # Compute new heuristic
                    new_h_cost, new_solution_vector = self._compute_marking_equation(new_marking)
                    have_exact_known_solution = True
    
                new_node = AStarSearchNode(
                    marking=new_marking,
                    g_cost=new_g_cost,
                    h_cost=new_h_cost,
                    parent=current_node,
                    transition_to_ancestor=transition,
                    total_model_moves=current_node.total_model_moves + (1 if transition.move_type in {'model', 'sync'} else 0),
                    solution_vector=new_solution_vector,
                    have_exact_known_solution=have_exact_known_solution
                )
    
                reached_set[new_marking.places] = new_g_cost
                heappush(open_set, new_node)
    
        return None, float('inf'), nodes_popped  # No path found

    # Correct working final code 12/08/2024    
    # def _compute_marking_equation(self, marking: Any) -> Tuple[float, np.ndarray]:
    #     """
    #     Compute the heuristic based on the marking equation, considering transition weights.
    #     Uses linear programming to minimize c^T x where c is the transition cost vector and x is the solution vector,
    #     subject to Ax = b and x >= 0, where A is the incidence matrix and b is the marking difference.
    #     Returns both the heuristic value and the solution vector.
    #     Raises an exception if no solution is found.
    #     """
    #     marking_diff = np.array(self.final_mark.places) - np.array(marking.places)
    #     incidence_matrix = self._incidence_matrix
    
    #     # Create the cost vector c (transition weights)
    #     c = np.array([transition.weight for transition in self.transitions])
    
    #     # Set up the bounds for x (all non-negative)
    #     bounds = [(0, None) for _ in range(len(self.transitions))]
    
    #     # Solve the linear programming problem
    #     result = linprog(c, A_eq=incidence_matrix, b_eq=marking_diff, bounds=bounds, method='highs')
    
    #     if result.success:
    #         cost = result.fun  # This is already c^T x
    #         return cost, result.x
    #     else:
    #         raise ValueError("Failed to find a solution for the heuristic computation.")

    def _compute_marking_equation(self, marking: Any) -> Tuple[float, np.ndarray]:
        """
        Compute the heuristic based on the marking equation, considering transition weights.
        Uses Gurobi if available, otherwise falls back to SciPy.
        
        Returns both the heuristic value and the solution vector.
        Raises an exception if no solution is found by either method.
        """
        marking_diff = np.array(self.final_mark.places) - np.array(marking.places)
        incidence_matrix = self._incidence_matrix
        c = np.array([transition.weight for transition in self.transitions])

        if GUROBI_AVAILABLE:
            try:
                return self._solve_with_gurobi(c, incidence_matrix, marking_diff)
            except Exception as e:
                print(f"Gurobi optimization failed: {e}. Falling back to SciPy.")

        # Fallback to SciPy linprog
        return self._solve_with_scipy(c, incidence_matrix, marking_diff)

    def _solve_with_gurobi(self, c: np.ndarray, incidence_matrix: np.ndarray, marking_diff: np.ndarray) -> Tuple[float, np.ndarray]:
        if not GUROBI_AVAILABLE:
            raise ImportError("Gurobi is not available")

        with self.gp.Env(empty=True) as env:
            env.setParam('OutputFlag', 0)
            env.start()
            with self.gp.Model(env=env) as model:
                num_transitions = len(c)
                x = model.addMVar(shape=num_transitions, lb=0, vtype=self.GRB.CONTINUOUS, name="x")
                model.setObjective(c @ x, self.GRB.MINIMIZE)
                model.addConstr(incidence_matrix @ x == marking_diff, name="eq_constraints")
                model.optimize()

                if model.status == self.GRB.OPTIMAL:
                    return model.objVal, x.X
                else:
                    raise ValueError("Gurobi failed to find an optimal solution.")

    def _solve_with_scipy(self, c: np.ndarray, incidence_matrix: np.ndarray, marking_diff: np.ndarray) -> Tuple[float, np.ndarray]:
        bounds = [(0, None) for _ in range(len(c))]
        result = linprog(c, A_eq=incidence_matrix, b_eq=marking_diff, bounds=bounds, method='highs')

        if result.success:
            return result.fun, result.x
        else:
            raise ValueError("SciPy failed to find a solution for the heuristic computation.")


    
    # correct working solution - without vector solution reusing
    # def astar_search(self, prob_dict: Optional[Dict[Any, Any]] = None, 
    #                  lamda: float = 0.5, 
    #                  trace_recovery: bool = False,
    #                  max_hist_len: Optional[int] = None) -> Tuple[List[Any], float]:
    #     """
    #     A* search algorithm for finding the optimal alignment in a Petri Net synchronous product.
    #     Args:
    #         prob_dict: Dictionary for conditioned probability computation.
    #         lamda: Weight factor for balancing between model and log moves.
    #         trace_recovery: Whether to recover the trace.
    #         max_hist_len: Maximum history length for conditioned probability computation.
    #     Returns:
    #         A tuple containing the optimal alignment (list of transitions) and its cost.
    #     """
    #     # Compute incidence matrix if not already computed
    #     if self._incidence_matrix is None:
    #         self._incidence_matrix = self._get_incidence_matrix()
    
    #     open_set = []
    #     heapify(open_set)
    #     closed_set = {}  # Dictionary to store best global cost for each marking
    #     reached_set = {}  # Dictionary to store temporary best known cost for each marking
        
    #     initial_node = AStarSearchNode(
    #         marking=self.init_mark,
    #         g_cost=0,
    #         h_cost=self._compute_heuristic(self.init_mark)
    #     )
    #     heappush(open_set, initial_node)
        
    #     while open_set:
    #         current_node = heappop(open_set)
            
    #         if current_node.marking.places == self.final_mark.places:
    #             return self._process_final_node(current_node, trace_recovery)
            
    #         if current_node.marking.places in closed_set:
    #             continue
            
    #         closed_set[current_node.marking.places] = current_node.g_cost
            
    #         for transition in self._find_available_transitions(current_node.marking.places):
    #             new_marking = self._fire_transition(current_node.marking, transition)
                
    #             transition_cost = self.compute_conditioned_weight(
    #                 current_node.path_prefix,
    #                 transition,
    #                 prob_dict, 
    #                 max_hist_len,
    #                 lamda=lamda
    #             )
                
    #             new_g_cost = current_node.g_cost + transition_cost
                
    #             # Check if the new marking has been reached with a lower or equal cost, or if it's already in the closed set
    #             if (new_marking.places in closed_set or 
    #                 (new_marking.places in reached_set and reached_set[new_marking.places] <= new_g_cost)):
    #                 continue
                
    #             new_h_cost = self._compute_heuristic(new_marking)
                
    #             new_path_prefix = (
    #                 current_node.path_prefix + [transition] if prob_dict is not None 
    #                 else []
    #             )
                
    #             new_node = AStarSearchNode(
    #                 marking=new_marking,
    #                 g_cost=new_g_cost,
    #                 h_cost=new_h_cost,
    #                 path_prefix=new_path_prefix,
    #                 parent=current_node,
    #                 transition_to_ancestor=transition,
    #                 total_model_moves=current_node.total_model_moves + (1 if transition.move_type in {'model', 'sync'} else 0)
    #             )
                
    #             reached_set[new_marking.places] = new_g_cost
    #             heappush(open_set, new_node)
        
    #     return None, float('inf')  # No path found

    
    def _process_final_node(self, node, nodes_popped=0, trace_recovery=False):
        """
        Process the final node in the search algorithm, returning the path, cost, and number of nodes popped.
    
        Parameters:
            node: The final node (can be of type AStarSearchNode, AStarIncrementalSearchNode, or ReachNode).
            nodes_popped (int): The number of nodes popped during the search.
            trace_recovery (bool): Whether to return only the reconstructed path.
    
        Returns:
            Tuple: (The reconstructed path, the cost, and the number of nodes popped).
        """
        path = self._reconstruct_path(node)
        
        if trace_recovery:
            return path
        else:
            if isinstance(node, ReachNode):
                return path, node.c, nodes_popped
            else:  # AStarSearchNode or AStarIncrementalSearchNode
                return path, node.g_cost, nodes_popped


    # correct working solution - without vector solution reusing    
    # def _compute_heuristic(self, marking: Any) -> float:
    #     """
    #     Compute the heuristic based on the marking equation, considering transition weights.
    #     Uses linear programming to minimize c^T x where c is the transition cost vector and x is the solution vector,
    #     subject to Ax = b and x >= 0, where A is the incidence matrix and b is the marking difference.
    #     Raises an exception if no solution is found.
    #     """
    #     marking_diff = np.array(self.final_mark.places) - np.array(marking.places)
    #     incidence_matrix = self._incidence_matrix
    
    #     # Create the cost vector c (transition weights)
    #     c = np.array([transition.weight for transition in self.transitions])
    
    #     # Set up the bounds for x (all non-negative)
    #     bounds = [(0, None) for _ in range(len(self.transitions))]
    
    #     # Solve the linear programming problem
    #     result = linprog(c, A_eq=incidence_matrix, b_eq=marking_diff, bounds=bounds, method='highs')
    
    #     if result.success:
    #         cost = result.fun  # This is already c^T x
    #         return cost
    #     else:
    #         raise ValueError("Failed to find a solution for the heuristic computation.")

    
    def _get_incidence_matrix(self):
        """Compute and return the incidence matrix of the Petri net."""
        num_places = len(self.places)
        num_transitions = len(self.transitions)
        incidence_matrix = np.zeros((num_places, num_transitions))
        
        for t_idx, transition in enumerate(self.transitions):
            for arc in transition.in_arcs:
                p_idx = self.places_indices[arc.source.name]
                incidence_matrix[p_idx, t_idx] = -1
            for arc in transition.out_arcs:
                p_idx = self.places_indices[arc.target.name]
                incidence_matrix[p_idx, t_idx] = 1
        
        return incidence_matrix


    def _get_consumption_matrix(self):
        num_places = len(self.places)
        num_transitions = len(self.transitions)
        
        # Create a zero matrix of size |P| x |T|
        C_minus = np.zeros((num_places, num_transitions))
        
        # Iterate through all arcs
        for arc in self.arcs:
            if isinstance(arc.source, Place) and isinstance(arc.target, Transition):
                i = self.places_indices[arc.source.name]
                j = self.transitions_indices[arc.target.name]
                C_minus[i, j] = -1
        
        return C_minus
        
    
    def _reconstruct_path(self, node) -> List[Any]:
        """
        Reconstruct the path from the start node to the given node.
    
        This function supports nodes of types AStarSearchNode, AStarIncrementalSearchNode, and ReachNode.
    
        Parameters:
            node: The final node (can be of type AStarSearchNode, AStarIncrementalSearchNode, or ReachNode).
    
        Returns:
            List[Any]: The reconstructed path from the start node to the given node.
        """
        if isinstance(node, ReachNode):
            return node.gamma  # Return the full alignment directly
    
        path = []
        while node.parent is not None:
            if isinstance(node, (AStarSearchNode, AStarIncrementalSearchNode)):
                path.append(node.transition_to_ancestor)
            node = node.parent
        return path[::-1]


    # Correct working code 12/08/2024
    # def _compute_extended_marking_equation(self, k_set):
    #     if k_set is None:
    #         k_set = set()

    #     # Compute c from transition weights and save it for future use
    #     self.c = [t.weight for t in self.transitions]
        
    #     model = LpProblem("Heuristic-Estimator", LpMinimize)
        
    #     # Define variables
    #     n_ys = len(k_set)
    #     n_xs = n_ys + 1
        
    #     X_variables = LpVariable.dicts("X", 
    #                                    ((i, j) for i in range(n_xs) for j in range(self._incidence_matrix.shape[1])), 
    #                                    lowBound=0)
    #     Y_variables = LpVariable.dicts("Y", 
    #                                    ((i, j) for i in range(n_ys) for j in range(self._incidence_matrix.shape[1])), 
    #                                    lowBound=0, upBound=1)
        
    #     # Objective Function
    #     obj_func = lpSum(self.c[j] * X_variables[i, j] for i in range(n_xs) for j in range(self._incidence_matrix.shape[1]))
    #     obj_func += lpSum(self.c[j] * Y_variables[i, j] for i in range(n_ys) for j in range(self._incidence_matrix.shape[1]))
    #     model += obj_func
        
    #     # Constraint 1
    #     for i in range(self._incidence_matrix.shape[0]):
    #         constraint_1 = (
    #             self.init_mark.places[i] + 
    #             lpSum(self._incidence_matrix[i, j] * X_variables[a, j] 
    #                   for a in range(n_xs) for j in range(self._incidence_matrix.shape[1])) +
    #             lpSum(self._incidence_matrix[i, j] * Y_variables[a, j] 
    #                   for a in range(n_ys) for j in range(self._incidence_matrix.shape[1])) 
    #             == self.final_mark.places[i]
    #         )
    #         model += constraint_1
        
    #     # Constraint 2
    #     for a in range(1, n_ys+1):
    #         for i in range(self._incidence_matrix.shape[0]):
    #             constraint_2 = (
    #                 self.init_mark.places[i] + 
    #                 lpSum(self._incidence_matrix[i, j] * X_variables[b, j] 
    #                       for b in range(a) for j in range(self._incidence_matrix.shape[1])) + 
    #                 lpSum(self._incidence_matrix[i, j] * Y_variables[b, j] 
    #                       for b in range(a-1) for j in range(self._incidence_matrix.shape[1])) + 
    #                 lpSum(self._consumption_matrix[i, j] * Y_variables[a-1, j] 
    #                       for j in range(self._incidence_matrix.shape[1])) >= 0
    #             )
    #             model += constraint_2
        
    #     # Constraint 5
    #     for a, trace_location_idx in enumerate(sorted(list(k_set)), start=1):
    #         for j in range(len(self.transitions)):
    #             if (self.transitions[j].move_type not in {'sync', 'trace'} or 
    #                     self.transitions[j].label != self.trace_model.transitions[trace_location_idx].label):
    #                 constraint_5 = Y_variables[a-1, j] == 0
    #                 model += constraint_5
        
    #     # Constraint 6
    #     for a in range(n_ys):
    #         constraint_6 = lpSum(Y_variables[a, j] for j in range(self._incidence_matrix.shape[1])) == 1
    #         model += constraint_6
        
    #     # Solve the model
    #     model.solve(PULP_CBC_CMD(msg=False))
        
    #     if LpStatus[model.status] != 'Optimal':
    #         raise ValueError("No optimal solution found.")
        
    #     # Extract solution
    #     sol_vec = np.zeros(self._incidence_matrix.shape[1])
    #     for i in range(n_xs):
    #         for j in range(self._incidence_matrix.shape[1]):
    #             sol_vec[j] += X_variables[i, j].varValue
    #     for i in range(n_ys):
    #         for j in range(self._incidence_matrix.shape[1]):
    #             sol_vec[j] += Y_variables[i, j].varValue
        
    #     heuristic_distance = model.objective.value()
        
    #     return heuristic_distance, sol_vec


    def _compute_extended_marking_equation(self, k_set):
        if k_set is None:
            k_set = set()
        
        # Compute c from transition weights and save it for future use
        self.c = [t.weight for t in self.transitions]
        
        if GUROBI_AVAILABLE:
            try:
                return self._solve_with_gurobi_extended(k_set)
            except gp.GurobiError as e:
                print(f"Gurobi optimization failed: {e}. Falling back to PuLP.")
            except Exception as e:
                print(f"Unexpected error with Gurobi: {e}. Falling back to PuLP.")

        return self._solve_with_pulp(k_set)

    def _solve_with_gurobi_extended(self, k_set):
        n_ys = len(k_set)
        n_xs = n_ys + 1

        with gp.Env(empty=True) as env:
            env.setParam('OutputFlag', 0)
            env.start()
            with gp.Model(env=env) as model:
                # Define variables
                X_variables = model.addMVar((n_xs, self._incidence_matrix.shape[1]), lb=0, vtype=GRB.CONTINUOUS, name="X")
                Y_variables = model.addMVar((n_ys, self._incidence_matrix.shape[1]), lb=0, ub=1, vtype=GRB.CONTINUOUS, name="Y")

                # Objective Function
                obj_func = gp.quicksum(self.c[j] * X_variables[i, j] for i in range(n_xs) for j in range(self._incidence_matrix.shape[1]))
                obj_func += gp.quicksum(self.c[j] * Y_variables[i, j] for i in range(n_ys) for j in range(self._incidence_matrix.shape[1]))
                model.setObjective(obj_func, GRB.MINIMIZE)

                # Constraint 1
                model.addConstrs(
                    (self.init_mark.places[i] + 
                    gp.quicksum(self._incidence_matrix[i, j] * X_variables[a, j] for a in range(n_xs) for j in range(self._incidence_matrix.shape[1])) +
                    gp.quicksum(self._incidence_matrix[i, j] * Y_variables[a, j] for a in range(n_ys) for j in range(self._incidence_matrix.shape[1])) 
                    == self.final_mark.places[i] for i in range(self._incidence_matrix.shape[0]))
                )

                # Constraint 2
                model.addConstrs(
                    (self.init_mark.places[i] + 
                    gp.quicksum(self._incidence_matrix[i, j] * X_variables[b, j] for b in range(a) for j in range(self._incidence_matrix.shape[1])) + 
                    gp.quicksum(self._incidence_matrix[i, j] * Y_variables[b, j] for b in range(a-1) for j in range(self._incidence_matrix.shape[1])) + 
                    gp.quicksum(self._consumption_matrix[i, j] * Y_variables[a-1, j] for j in range(self._incidence_matrix.shape[1])) >= 0
                    for a in range(1, n_ys+1) for i in range(self._incidence_matrix.shape[0]))
                )

                # Constraint 5
                for a, trace_location_idx in enumerate(sorted(list(k_set)), start=1):
                    for j in range(len(self.transitions)):
                        if (self.transitions[j].move_type not in {'sync', 'trace'} or 
                                self.transitions[j].label != self.trace_model.transitions[trace_location_idx].label):
                            model.addConstr(Y_variables[a-1, j] == 0)

                # Constraint 6
                model.addConstrs((gp.quicksum(Y_variables[a, j] for j in range(self._incidence_matrix.shape[1])) == 1 for a in range(n_ys)))

                # Solve the model
                model.optimize()

                if model.status == GRB.OPTIMAL:
                    # Extract solution
                    sol_vec = X_variables.X.sum(axis=0) + Y_variables.X.sum(axis=0)
                    heuristic_distance = model.objVal
                    return heuristic_distance, sol_vec
                else:
                    raise gp.GurobiError("Gurobi failed to find an optimal solution.")


    def _solve_with_pulp(self, k_set):
        # Original PuLP-based implementation
        model = LpProblem("Heuristic-Estimator", LpMinimize)
        
        # Define variables
        n_ys = len(k_set)
        n_xs = n_ys + 1
        
        X_variables = LpVariable.dicts("X", 
                                       ((i, j) for i in range(n_xs) for j in range(self._incidence_matrix.shape[1])), 
                                       lowBound=0)
        Y_variables = LpVariable.dicts("Y", 
                                       ((i, j) for i in range(n_ys) for j in range(self._incidence_matrix.shape[1])), 
                                       lowBound=0, upBound=1)
        
        # Objective Function
        obj_func = lpSum(self.c[j] * X_variables[i, j] for i in range(n_xs) for j in range(self._incidence_matrix.shape[1]))
        obj_func += lpSum(self.c[j] * Y_variables[i, j] for i in range(n_ys) for j in range(self._incidence_matrix.shape[1]))
        model += obj_func
        
        # Constraint 1
        for i in range(self._incidence_matrix.shape[0]):
            constraint_1 = (
                self.init_mark.places[i] + 
                lpSum(self._incidence_matrix[i, j] * X_variables[a, j] 
                      for a in range(n_xs) for j in range(self._incidence_matrix.shape[1])) +
                lpSum(self._incidence_matrix[i, j] * Y_variables[a, j] 
                      for a in range(n_ys) for j in range(self._incidence_matrix.shape[1])) 
                == self.final_mark.places[i]
            )
            model += constraint_1
        
        # Constraint 2
        for a in range(1, n_ys+1):
            for i in range(self._incidence_matrix.shape[0]):
                constraint_2 = (
                    self.init_mark.places[i] + 
                    lpSum(self._incidence_matrix[i, j] * X_variables[b, j] 
                          for b in range(a) for j in range(self._incidence_matrix.shape[1])) + 
                    lpSum(self._incidence_matrix[i, j] * Y_variables[b, j] 
                          for b in range(a-1) for j in range(self._incidence_matrix.shape[1])) + 
                    lpSum(self._consumption_matrix[i, j] * Y_variables[a-1, j] 
                          for j in range(self._incidence_matrix.shape[1])) >= 0
                )
                model += constraint_2
        
        # Constraint 5
        for a, trace_location_idx in enumerate(sorted(list(k_set)), start=1):
            for j in range(len(self.transitions)):
                if (self.transitions[j].move_type not in {'sync', 'trace'} or 
                        self.transitions[j].label != self.trace_model.transitions[trace_location_idx].label):
                    constraint_5 = Y_variables[a-1, j] == 0
                    model += constraint_5
        
        # Constraint 6
        for a in range(n_ys):
            constraint_6 = lpSum(Y_variables[a, j] for j in range(self._incidence_matrix.shape[1])) == 1
            model += constraint_6
        
        # Solve the model
        model.solve(PULP_CBC_CMD(msg=False))
        
        if LpStatus[model.status] != 'Optimal':
            raise ValueError("No optimal solution found.")
        
        # Extract solution
        sol_vec = np.zeros(self._incidence_matrix.shape[1])
        for i in range(n_xs):
            for j in range(self._incidence_matrix.shape[1]):
                sol_vec[j] += X_variables[i, j].varValue
        for i in range(n_ys):
            for j in range(self._incidence_matrix.shape[1]):
                sol_vec[j] += Y_variables[i, j].varValue
        
        heuristic_distance = model.objective.value()
        
        return heuristic_distance, sol_vec

    
# astar_incremental can not be used for sk logs - constraint 5 has to be modified and generalize to allow several labels to fire!!!
    def astar_incremental(self, s=0, k=None, nodes_popped=0):
    
        if self._incidence_matrix is None:
            self._incidence_matrix = self._get_incidence_matrix()
        if self._consumption_matrix is None:
            self._consumption_matrix = self._get_consumption_matrix()
    
        open_set = []
        heapify(open_set)
        closed_set = set()
        reached_dict = {}
    
        if k is None:
            k = {0}
    
        initial_h_cost, initial_solution_vector = self._compute_extended_marking_equation(k_set=k)
        initial_node = AStarIncrementalSearchNode(
            marking=self.init_mark,
            g_cost=0,
            h_cost=initial_h_cost,
            solution_vector=initial_solution_vector,
            has_exact_heuristic=True
        )
        heappush(open_set, initial_node)
    
        while open_set:
            current_node = heappop(open_set)
            nodes_popped += 1
            
            if current_node.marking.places == self.final_mark.places:
                return self._process_final_node(current_node, nodes_popped)
    
            if current_node.marking.places in closed_set:
                continue
    
            if not current_node.has_exact_heuristic:
                if s not in k and s < len(self.trace_model.transitions):
                    k.add(s)
                    return self.astar_incremental(s=0, k=k, nodes_popped=nodes_popped)
    
                new_h_cost, new_solution_vector = self._compute_marking_equation(current_node.marking)
    
                if new_h_cost > current_node.h_cost:
                    current_node.update_heuristic(new_h_cost, new_solution_vector)
                    heappush(open_set, current_node)
                    continue
    
                current_node.update_heuristic(new_h_cost, new_solution_vector)
    
            closed_set.add(current_node.marking.places)
            s = max(s, current_node.n_events_explained)
    
            for transition in self._find_available_transitions(current_node.marking.places):
                new_marking = self._fire_transition(current_node.marking, transition)
                new_g_cost = current_node.g_cost + transition.weight
    
                if (new_marking.places in closed_set or 
                    (new_marking.places in reached_dict and reached_dict[new_marking.places] <= new_g_cost)):
                    continue
    
                reached_dict[new_marking.places] = new_g_cost
    
                if current_node.solution_vector[self.transitions_indices[transition.name]] >= 1:
                    # Reuse solution
                    new_h_cost = current_node.h_cost - transition.weight
                    new_solution_vector = current_node.solution_vector.copy()
                    new_solution_vector[self.transitions_indices[transition.name]] -= 1
                    has_exact_heuristic = True
                else:
                    # Compute estimated heuristic
                    new_h_cost = max(current_node.h_cost - transition.weight, 0)
                    new_solution_vector = None
                    has_exact_heuristic = False
    
                n_events_explained = current_node.n_events_explained + (1 if transition.move_type in {'trace', 'sync'} else 0)
         
                new_node = AStarIncrementalSearchNode(
                    marking=new_marking,
                    g_cost=new_g_cost,
                    h_cost=new_h_cost,
                    parent=current_node,
                    transition_to_ancestor=transition,
                    solution_vector=new_solution_vector,
                    has_exact_heuristic=has_exact_heuristic,
                    n_events_explained=n_events_explained
                )
                heappush(open_set, new_node)
    
        return None, float('inf'), nodes_popped  # No path found    

    
    def reach_search(self) -> Tuple[Optional[List[Any]], float]:
        """
        Perform the REACH search algorithm to find the first optimal alignment.
        
        Returns:
            A tuple containing the first optimal alignment (or None if not found) and its cost.
        """
        if self.net.mandatory_transitions_map is None:
            self.net.mandatory_transitions_map = compute_mandatory_transitions(
                self.net.pm4py_net, 
                self.net.pm4py_initial_marking, 
                self.net.pm4py_final_marking,
                self.net.place_mapping
            )
       
        if self.net.alive_transitions_map is None:
            self.net.alive_transitions_map = map_markings_to_reachable_transitions(self.net)       
    
        open_set = []
        visited = set()  # Set to keep track of visited markings
        best_known_cost = {}  # Dictionary to keep track of the best known cost to reach each state
        
        greedy_alignment, greedy_nodes_popped = self.align_greedy()
        max_cost = greedy_alignment.c if greedy_alignment else float('inf')
        
        initial_state = self.create_initial_state()
        heappush(open_set, initial_state)
        best_known_cost[self.state_id(initial_state)] = initial_state.c

        nodes_popped = greedy_nodes_popped
        
        while open_set:
            current_state = heappop(open_set)
            nodes_popped += 1
            
            state_id = self.state_id(current_state)
    
            if state_id in visited:
                continue
    
            visited.add(state_id)
    
            if self.is_final(current_state):
                return self._process_final_node(current_state, nodes_popped)
            
            self.add_neighbors(current_state, open_set, max_cost, visited, best_known_cost)
    
        # If no alignment was found, return None and infinity as the cost
        return None, float('inf'), nodes_popped
    
    def state_id(self, state: ReachNode) -> Tuple[Tuple[int, ...], int]:
        return state.marking.places
        
    def create_initial_state(self) -> ReachNode:
        return ReachNode(marking=self.init_mark, i=0, gamma=[], c=0, h=self.reach_heuristic(self.init_mark, 0))

    def add_neighbors(self, state: ReachNode, open_set: List[ReachNode], max_cost: float, 
                      visited: Set[Tuple[int, ...]], 
                      best_known_cost: Dict[Tuple[int, ...], float]) -> None:
        
        enabled_transitions = self._find_available_transitions(state.marking.places)
        sync_transitions = [t for t in enabled_transitions if t.move_type == 'sync']
        
        if sync_transitions:
            transitions_to_add = enabled_transitions
        elif self.less_states_trace(state):
            transitions_to_add = [t for t in enabled_transitions if t.move_type == 'model']
        elif self.less_states_model(state):
            transitions_to_add = [t for t in enabled_transitions if t.move_type == 'trace']
        else:
            transitions_to_add = enabled_transitions
        
        for transition in transitions_to_add:
            new_state = self.create_new_state(state, transition)
            new_state_id = self.state_id(new_state)
                       
            if new_state_id not in visited and new_state.f_cost <= max_cost:
                if new_state_id not in best_known_cost or new_state.c < best_known_cost[new_state_id]:
                    best_known_cost[new_state_id] = new_state.c
                    heappush(open_set, new_state)

    def add_move(self, parent_state: ReachNode, open_set: List[ReachNode], transition: Any, max_cost: float) -> None:
        new_state = self.create_new_state(parent_state, transition)
        
        if self.should_add(new_state, open_set, max_cost):
            heappush(open_set, new_state)

    def create_new_state(self, parent_state: ReachNode, transition: Any) -> ReachNode:
        new_marking = self._fire_transition(parent_state.marking, transition)
        
        if transition.move_type in ['sync', 'trace']:
            new_i = parent_state.i + 1
        elif transition.move_type == 'model':
            new_i = parent_state.i
        else:
            raise ValueError(f"Invalid move type: {transition.move_type}")
            
        new_gamma = parent_state.gamma + [transition.name]
        new_c = parent_state.c + transition.weight
        new_h = self.reach_heuristic(new_marking, new_i)
        
        return ReachNode(marking=new_marking, i=new_i, gamma=new_gamma, c=new_c, h=new_h)
 
    def reach_heuristic(self, marking: Any, i: int) -> float:

        # Get the required mandatory transitions from the model
        required_transitions = self.get_mandatory_transitions(marking)
        
        # Get the remaining transitions in the trace from the current index i onwards
        trace_transitions_remaining = [t.label for t in self.trace_model.transitions[i:]]
        
        # Identify the missing required transitions by checking those not in the remaining trace
        missing_required_transitions = required_transitions - set(trace_transitions_remaining)

        # Process transitions that are present in the trace
        trace_transitions_present = []
        for t in trace_transitions_remaining:
            if t not in missing_required_transitions:
                trace_transitions_present.append(t)

        # Calculate the heuristic value
        heuristic_value = len(missing_required_transitions) + len(trace_transitions_present) * self.net.epsilon

        return heuristic_value
    
    def get_mandatory_transitions(self, marking: Any) -> Set[str]:
        # Extract only the net model's part of the marking
        net_marking = Marking(marking.places[:len(self.net.places)])
        net_marking_tuple = net_marking.places
        return self.net.mandatory_transitions_map[net_marking_tuple]
    
    def less_states_trace(self, state: ReachNode) -> bool:
        enabled_transitions = self._find_available_transitions(state.marking.places)

        if not enabled_transitions:
            raise ValueError("No enabled transitions found in the given marking.")

        # Check for synchronous transitions
        if any(t.move_type == 'sync' for t in enabled_transitions):
            return False
        
        remaining_activities = set(t.label for t in self.trace_model.transitions[state.i:])
        model_transitions = [t for t in enabled_transitions if t.move_type == 'model']
        
        if not model_transitions:
            return False
            
        # Check if none of the enabled model move transitions' labels (where label is not None) appear in the remaining activities
        return all(
            t.label not in remaining_activities 
            for t in enabled_transitions 
            if t.move_type == 'model' 
        )

    def less_states_model(self, state: ReachNode) -> bool:

        if state.i >= len(self.trace_model.transitions):
            return False
        
        # Extract only the net model's part of the marking
        net_marking = state.marking.places[:len(self.net.places)]

        if net_marking not in self.net.alive_transitions_map:
            self.net.alive_transitions_map = map_markings_to_reachable_transitions(self.net)
        
        alive_activities = self.net.alive_transitions_map[net_marking]['reachable_transitions']
        return self.trace_model.transitions[state.i].label not in alive_activities

    def align_greedy(self) -> Optional[ReachNode]:
        state = self.create_initial_state()
        visited = set()
        nodes_popped = 0
    
        while state is not None:
            nodes_popped += 1 
            
            if self.is_final(state):
                return state, nodes_popped
    
            visited.add(tuple(state.marking.places))
            
            open_queue = self.get_neighbors(state)
            
            state = None
            while open_queue:
                candidate = min(open_queue, key=lambda x: x.c + x.h)
                open_queue.remove(candidate)
                
                if tuple(candidate.marking.places) not in visited:
                    state = candidate
                    break
    
        return None, nodes_popped
    
    def get_neighbors(self, state: 'ReachNode') -> List['ReachNode']:
        enabled_transitions = self._find_available_transitions(state.marking.places)
        neighbor_states = []

        for transition in enabled_transitions:
            neighbor_states.append(self.create_new_state(state, transition))
        
        return neighbor_states                     

    def should_add(self, state: ReachNode, open_set: List[ReachNode], max_cost: float) -> bool:
        return state.c + state.h < max_cost


    def is_final(self, state):
        return self.final_mark.places == state.marking.places

def generate_new_marking_for_new_window(curr_model_marking, trace_model):
    if curr_model_marking is None:
        return None
    
    trace_mark = tuple([1] + [0] * (len(trace_model.places) - 1))
    model_mark = curr_model_marking.places
    
    return Marking(model_mark + trace_mark)


def select_top_n_unique_markings(candidate_triplets_lst, n_unique_markings, last_window=False):
    candidate_triplets_lst.sort(key=lambda x: x[1] + x[4])
    
    # In the last window all markings are the same -- i.e., final place in the model net and thus taking best alignments with same final marking
    if last_window:
        return candidate_triplets_lst[:n_unique_markings]
    
    # Set to keep track of unique third elements
    unique_third_elements = set()

    # List to store the selected triplets
    alignment_dist_marking_triplets = []
    
    for triplet in candidate_triplets_lst:
        if triplet[2].places not in unique_third_elements:
            alignment_dist_marking_triplets.append(triplet)
            unique_third_elements.add(triplet[2].places)
        if len(alignment_dist_marking_triplets) == n_unique_markings:
            break

    return alignment_dist_marking_triplets


# def horizon_based_conformance(net, trace_df, window_len=3, cost_function=None, hist_prob_dict=None,
#                               lamda=None, partial_conformance=True, n_unique_final_markings=1,
#                               record_marking_ancestors=False, explor_reward=0.001, window_overlap=0,
#                               net_details_dict=None, reachability_heuristic=True):
    
#     if window_overlap >= len(trace_df):
#         raise ValueError(f'Overlap length is {window_overlap} and the trace length is {len(trace_df)} which is not allowed.')
    
    
#     if window_len is None:
#         window_len = len(trace_df)
    
#     total_nodes_opened = 0
#     alignment_hist = []
#     curr_model_marking = None
#     subtraces_lst = split_dataframe(trace_df, window_len, overlap=window_overlap) 
#     total_cost_lst = []
#     alignment_dist_marking_triplets = [(None, 0, None, 0)]
#     candidate_triplets = []
#     partial_conformance = True
#     ancestors_nodes_lst = []
#     ancestor_marking_node = MarkingAncestorNode(ancestor_marking_node=None, dist=0, alignment=None)
#     ancestors_nodes_lst.append(ancestor_marking_node)
#     trace_activities_multiset = {}
#     total_nodes_opened = 0
    
#     if reachability_heuristic and window_len is not None:
#         trace_activities_multiset = get_activity_multiset_single_trace(trace_df)
#         # print(f'Here is the activities left within the trace: {trace_activities_multiset} \n')
#         # print(f'Here are the net details dict: {net_details_dict} \n')
        
#     for idx, trace in enumerate(subtraces_lst):
#         # print(f'--------------------starting new subtrace--------------------------------')
#         print(f'Evaluating subtrace {idx+}/{len(subtraces_lst)}...', end='\r', flush=True)
#         # display(trace)
#         trace_model = construct_trace_model(trace)
#         # print(f'The length of the subtrace_lst={len(subtraces_lst)} and the idx={idx}')
#         if idx == len(subtraces_lst) - 1:
#             partial_conformance = False
#             reachability_heuristic = False
            
#         for triplet in alignment_dist_marking_triplets:
#             _, dist, curr_model_marking, ances_idx = triplet
#             curr_model_marking = generate_new_marking_for_new_window(curr_model_marking, trace_model)
#             sync_prod = SyncProduct(net, trace_model, cost_function, curr_model_marking)
#             search_triplets_lst, nodes_opened = sync_prod._dijkstra_no_rg_construct(prob_dict=None, lamda=lamda, 
#                                                                       partial_conformance=partial_conformance,
#                                                                       return_net_final_marking=True,
#                                                                       n_unique_final_markings=n_unique_final_markings,
#                                                                       explor_reward=explor_reward, overlap_size=window_overlap,
#                                                                       trace_activities_multiset=trace_activities_multiset, 
#                                                                       net_details_dict=net_details_dict,
#                                                                       use_heuristic_distance=reachability_heuristic)


#             candidate_triplets += [([elem.name for elem in t[0]], t[1] + dist, t[2], ances_idx, t[3]) for t in search_triplets_lst]
#             total_nodes_opened += nodes_opened
            
#         alignment_dist_marking_triplets = select_top_n_unique_markings(candidate_triplets, n_unique_final_markings,
#                                                                        last_window=not partial_conformance)
#         new_ancestors_nodes_lst = []
#         for t in alignment_dist_marking_triplets:
#             extended_alignment = ancestors_nodes_lst[t[3]].alignment + ['end'] + t[0]
#             new_node = MarkingAncestorNode(ancestor_marking_node=ancestors_nodes_lst[t[3]],
#                                            dist=t[1], alignment=extended_alignment)
#             new_ancestors_nodes_lst.append(new_node)
            
#         alignment_dist_marking_triplets = [(t[0], t[1], t[2], i) for i, t in enumerate(alignment_dist_marking_triplets)]  

#         ancestors_nodes_lst = new_ancestors_nodes_lst
#         candidate_triplets = []
#         trace_activities_multiset = subtract_activities(trace_activities_multiset, trace)
        
#     # Returning total distance of the shortest window-based path and the alignment
#     return alignment_dist_marking_triplets[0][1], ancestors_nodes_lst[0].alignment, total_nodes_opened


def horizon_based_conformance(net, trace_df, window_len=3, cost_function=None, hist_prob_dict=None,
                              lamda=0.5, partial_conformance=True, n_unique_final_markings=1,
                              record_marking_ancestors=False, explor_reward=0.001, window_overlap=0,
                              net_details_dict=None, use_heuristics=False):
    
    if window_overlap >= len(trace_df):
        raise ValueError(f'Overlap length is {window_overlap} and the trace length is {len(trace_df)} which is not allowed.')
    
    if window_len is None:
        window_len = len(trace_df)
    
    total_nodes_opened = 0
    alignment_hist = []
    curr_model_marking = None
    subtraces_lst = split_dataframe(trace_df, window_len, overlap=window_overlap)
    total_cost_lst = []
    alignment_dist_marking_triplets = [(None, 0, None, 0)]
    candidate_triplets = []
    partial_conformance = True
    ancestors_nodes_lst = []
    ancestor_marking_node = MarkingAncestorNode(ancestor_marking_node=None, dist=0, alignment=None)
    ancestors_nodes_lst.append(ancestor_marking_node)
    trace_activities_multiset = {}
    total_nodes_opened = 0


    max_hist_len = max((len(key) for key in hist_prob_dict.keys()), default=0) if hist_prob_dict else None
    
    if use_heuristics and window_len is not None:
        trace_activities_multiset = get_activity_multiset_single_trace(trace_df)
        
    for idx, trace in enumerate(subtraces_lst):
        trace_model = construct_trace_model(trace)
        
        is_last_subtrace = (idx == len(subtraces_lst) - 1)
        
        if is_last_subtrace:
            partial_conformance = False
            use_heuristics = False
            
        for triplet in alignment_dist_marking_triplets:
            _, dist, curr_model_marking, ances_idx = triplet
            curr_model_marking = generate_new_marking_for_new_window(curr_model_marking, trace_model)
            sync_prod = SyncProduct(net, trace_model, cost_function, curr_model_marking)
            
            search_triplets_lst, nodes_opened = sync_prod._dijkstra_no_rg_construct(
                prob_dict=hist_prob_dict,
                lamda=lamda, 
                partial_conformance=partial_conformance,
                return_net_final_marking=True,
                n_unique_final_markings=n_unique_final_markings,
                explor_reward=explor_reward,
                overlap_size=window_overlap,
                trace_activities_multiset=trace_activities_multiset, 
                use_heuristic_distance=use_heuristics,
                max_hist_len=max_hist_len
            )

            candidate_triplets += [(t[0], t[1] + dist, t[2], ances_idx, t[3]) for t in search_triplets_lst]
            total_nodes_opened += nodes_opened
            
        alignment_dist_marking_triplets = select_top_n_unique_markings(candidate_triplets, n_unique_final_markings,
                                                                       last_window=is_last_subtrace)
        new_ancestors_nodes_lst = []
        for t in alignment_dist_marking_triplets:
            extended_alignment = ancestors_nodes_lst[t[3]].alignment + ['end'] + t[0]
            new_node = MarkingAncestorNode(ancestor_marking_node=ancestors_nodes_lst[t[3]],
                                           dist=t[1], alignment=extended_alignment)
            new_ancestors_nodes_lst.append(new_node)
            
        alignment_dist_marking_triplets = [(t[0], t[1], t[2], i) for i, t in enumerate(alignment_dist_marking_triplets)]  

        ancestors_nodes_lst = new_ancestors_nodes_lst
        candidate_triplets = []
        trace_activities_multiset = subtract_activities(trace_activities_multiset, trace)
        
    # Returning total distance of the shortest window-based path and the alignment
    return alignment_dist_marking_triplets[0][1], ancestors_nodes_lst[0].alignment, total_nodes_opened
    

def get_unique_traces(df, min_len=None, max_len=None, limit=None):
    # Group by 'case:concept:name' and aggregate 'concept:name' into lists
    df_grouped = df.groupby('case:concept:name')['concept:name'].apply(list).reset_index(name='traces')
    
    # Filter traces based on their length if min_len and max_len are specified
    if min_len is not None:
        df_grouped = df_grouped[df_grouped['traces'].apply(lambda x: len(x) >= min_len)]
    if max_len is not None:
        df_grouped = df_grouped[df_grouped['traces'].apply(lambda x: len(x) <= max_len)]

    # Drop duplicate traces
    df_grouped = df_grouped.drop_duplicates(subset='traces')

    # If limit is specified, return only the top 'limit' traces
    if limit is not None:
        df_grouped = df_grouped.head(limit)

    # Expand the grouped dataframe back to the original format
    df = df_grouped.explode('traces')
    df.columns = ['case:concept:name', 'concept:name']

    return df

  

def sample_traces(df, n=1, random_seed=None):
    # Set the random seed if specified
    if random_seed is not None:
        np.random.seed(random_seed)

    # Group by 'case:concept:name' and aggregate 'concept:name' into lists
    df_grouped = df.groupby('case:concept:name')['concept:name'].apply(list).reset_index(name='traces')

    # Sample 'n' traces
    df_sampled = df_grouped.sample(n)

    # Expand the sampled dataframe back to the original format
    df = df_sampled.explode('traces')
    df.columns = ['case:concept:name', 'concept:name']

    return df


def sample_random_traces(df, n_traces=1, random_seed=None):
    """
    Sample traces from the log dataframe.
    
    Parameters:
    df (pd.DataFrame): The log dataframe with 'case:concept:name' and 'concept:name' columns.
    n_traces (int): The number of traces to sample. Default is 1.
    random_seed (int): Seed for random sampling. Default is None.
    
    Returns:
    pd.DataFrame: A dataframe containing the sampled traces.
    """
    if random_seed is not None:
        random.seed(random_seed)
    
    # Get unique case IDs
    case_ids = df['case:concept:name'].unique()
    
    # Randomly sample n_traces case IDs
    sampled_case_ids = random.sample(list(case_ids), min(n_traces, len(case_ids)))
    
    # Filter the dataframe to include only the sampled traces
    sampled_df = df[df['case:concept:name'].isin(sampled_case_ids)]
    
    return sampled_df
    

def prepare_process_model(df_train, non_sync_penalty=1, cost_function=None, return_markings=True, return_pm4py_model=True):
    
    pm4py_net, init_marking, final_marking = discover_process_model(df_train, return_markings=return_markings)
    
    if cost_function is None:
        def cost_function(x):
            if x != 0 or x != 1:
                raise ValueError("You are using the default cost function. x must be between 0 or 1")
            if x == 0:
                return 1
            return 0
   
    net_model = from_discovered_model_to_PetriNet(pm4py_net, non_sync_move_penalty=non_sync_penalty,
                                                  cost_function=cost_function, conditioned_prob_compute=False)
    
    if return_pm4py_model:
        return net_model, init_marking, final_marking, pm4py_net
    else:
        return net_model, init_marking, final_marking


# def compare_window_based_baselines(df, n_train_traces=None, n_test_traces=None, train_traces=None, test_traces=None,
#                                    window_lengths_lst=None, n_final_markings_lst=None, 
#                                    seed_1=101, seed_2=202, as_df=True, viz_proc_tree=False,
#                                    return_alignment_res_dict=False, return_model=False,
#                                    explor_reward=0.000, window_overlap=0, read_model_from_file=False, file_path='',
#                                    activity_mapping_dict=None, use_trace_frequencies=False, return_agg_res_dict=False,
#                                    reachability_heuristic=False, trace_recovery=False, map_activities_to_strings=True):

#     cost_res_dict = defaultdict(list)
#     alignment_res_dict = defaultdict(list)
#     time_res_dict = defaultdict(list)
#     nodes_opened_dict = defaultdict(list)

#     # Default values
#     if window_lengths_lst is None:
#         window_lengths_lst = [3]
#     if n_final_markings_lst is None:
#         n_final_markings_lst = [1]

#     # Handling the case when specific traces are provided
#     if read_model_from_file:
#         df_train = None

#     elif use_trace_frequencies and train_traces is None and n_train_traces is None:
#         # print('Using all the unique traces in the log to construct the process model')
#         # trace_frequencies = df.groupby(['case:concept:name'])['concept:name'].apply(list).value_counts().to_dict()
#         df_copy = df.copy()
#         # Generate a string representation of the trace for each case in the copied dataframe
#         df_copy['trace_str'] = df_copy.groupby('case:concept:name')['concept:name'].transform(lambda x: '->'.join(x))
#         # Identify unique traces by selecting one case ID per unique trace pattern
#         unique_trace_ids = df_copy.drop_duplicates('trace_str')['case:concept:name'].tolist()
#         # Use the original dataframe to filter for unique traces, ensuring the original format is maintained
#         df_train = df[df['case:concept:name'].isin(unique_trace_ids)].copy()

#     elif train_traces is not None:
#         df_train = df[df['case:concept:name'].isin(train_traces)]
#     else:
#         if n_train_traces is None:
#             n_train_traces = 15
#         # print(f'n_train traces for df_train={n_train_traces}')
#         df_train = sample_traces(df, n_train_traces, seed_1)

#     if test_traces is not None:
#         df_test = df[df['case:concept:name'].isin(test_traces)]

#     elif use_trace_frequencies and test_traces is None and n_test_traces is None:     
#         df_test = df[df['case:concept:name'].isin(unique_trace_ids)].copy()

#     else:
#         if n_test_traces is None:
#             n_test_traces = df['case:concept:name'].nunique()
#         # print(f'selecting {n_test_traces} traces for testing')
#         df_test = sample_traces(df, n_test_traces, seed_2)

#     if read_model_from_file:
#         net, init_marking, final_marking = generate_model_from_file(file_path, return_markings=True, activity_mapping_dict=activity_mapping_dict)
#         pm4py_model = net
#     else:
#         net, init_marking, final_marking, pm4py_model = prepare_process_model(df_train, return_pm4py_model=return_model)

#     if map_activities_to_strings:
#         df_test = prepare_determ_df_for_conformance(df_test)

#     if viz_proc_tree:
#         discover_process_tree(df_train)

#     cases = df_test['case:concept:name'].unique()

#     if reachability_heuristic:
#         print(f'Started computing net_details_dict')
#         net_details_dict = map_markings_to_reachable_transitions(net)
#         print(f'Finished computing net_details_dict')
#     else:
#         net_details_dict = None


#     for window_len in window_lengths_lst:

#         for n_markings in n_final_markings_lst:

#             print(f'Evaluating variant: n_markings={n_markings}, window_len={window_len}')

#             for case in cases:
#                 trace_df = df_test[df_test['case:concept:name'] == case]
#                 start_time = time.time()
#                 # display(trace_df)
#                 dist, full_alignment, nodes_opened = horizon_based_conformance(net, trace_df,
#                                                                  window_len=window_len,
#                                                                  n_unique_final_markings=n_markings,
#                                                                  explor_reward=explor_reward,
#                                                                  window_overlap=window_overlap,
#                                                                  net_details_dict=net_details_dict,
#                                                                  reachability_heuristic=reachability_heuristic)


#                 end_time = time.time()
#                 computation_time = end_time - start_time

#                 nodes_opened_dict[(n_markings, window_len)].append(nodes_opened)
#                 cost_res_dict[(n_markings, window_len)].append(dist)
#                 alignment_res_dict[(n_markings, window_len)].append(full_alignment + [dist])
#                 time_res_dict[(n_markings, window_len)].append(computation_time)  

#             if window_len is None:
#                 break


#     if as_df:
#         # Convert cost and time results into DataFrames
#         res_df = defaultdict_to_dataframe(cost_res_dict)
#         time_df = defaultdict_to_dataframe(time_res_dict)
#         nodes_opened_df = defaultdict_to_dataframe(nodes_opened_dict)

#         # Rename time_df columns to include '_time' suffix
#         time_df.columns = [f'{col.replace("_cost", "")}_time' for col in time_df.columns]
#         nodes_opened_df.columns = [f'{col.replace("_cost", "")}_nodes_opened' for col in nodes_opened_df.columns]

#         # Merge the cost DataFrame and time DataFrame
#         final_df = pd.concat([res_df, time_df, nodes_opened_df], axis=1)

#         for col in final_df.columns:
#             if '_time' in col:
#                 final_df[col] = final_df[col].astype(float).round(3)  # Format time columns
#             elif '_nodes_opened' in col:
#                 final_df[col] = final_df[col].astype(float).round(2)  # Format nodes opened columns
#             else:
#                 final_df[col] = final_df[col].astype(float).round(2)  # Format cost columns

#         # Assuming 'cases' list is predefined with the unique case IDs
#         if 'case:concept:name' not in final_df.columns:
#             final_df.insert(0, 'case:concept:name', cases.tolist())

#         # Calculate and add relative time columns for multiple markings compared to one marking
#         for window_len in window_lengths_lst:
#             one_marking_time_col = f'marking=1, window_len={window_len}_time'

#             for n_markings in n_final_markings_lst:
#                 if n_markings == 1:  # Skip the single marking case
#                     continue

#                 multiple_markings_time_col = f'marking={n_markings}, window_len={window_len}_time'
#                 # relative_time_col = f'Relative_Time_{n_markings}_vs_1_Markings_Window_{window_len}'

#                 # Check if the required columns exist to avoid KeyError
#                 # if one_marking_time_col in final_df.columns and multiple_markings_time_col in final_df.columns:
#                     # Calculate the relative time as percentage
#                     # final_df[relative_time_col] = (final_df[multiple_markings_time_col] / final_df[one_marking_time_col]) * 100
#                     # final_df[relative_time_col] = final_df[relative_time_col].round(2)  # Optional: Round the relative time


#         if use_trace_frequencies:
#             trace_frequencies = create_case_id_to_frequency_dict(df)
#             final_df['frequency'] = final_df['case:concept:name'].apply(lambda x: trace_frequencies.get(x, 0))

#             aggregated_results = {}

#             # Exclude the first column ('case:concept:name') from the aggregation process
#             columns_to_aggregate = final_df.columns[1:]  # Skip the first column

#             # Iterate over the columns to aggregate, calculating the weighted sum for each
#             for column in columns_to_aggregate:
#                 # Check if the column is not the 'frequency' column; adjust the condition as needed
#                 if column != 'frequency':
#                     # Calculate the weighted sum for the current column
#                     weighted_sum = (final_df[column] * final_df['frequency']).sum()
#                     # Store the result in the dictionary
#                     aggregated_results[column] = weighted_sum



#         if return_alignment_res_dict:
#             if return_model:
#                 if return_agg_res_dict:
#                     return final_df, alignment_res_dict, (pm4py_model, init_marking, final_marking), aggregated_results
#                 return final_df, alignment_res_dict, (pm4py_model, init_marking, final_marking)
#             if return_agg_res_dict:
#                 return final_df, alignment_res_dict, aggregated_results
#             return final_df, alignment_res_dict

#         if return_model:
#             if return_agg_res_dict:
#                 return final_df, (pm4py_model, init_marking, final_marking), aggregated_results
#             return final_df, (pm4py_model, init_marking, final_marking)

#         if return_agg_res_dict:
#             return final_df, aggregated_results

#         return final_df


def compare_window_based_baselines(df, sk_df=None, sftmax_lst=None, n_train_traces=None, n_test_traces=None,
                                   random_trace_selection=False, train_traces=None, test_traces=None,
                                   window_lengths_lst=None, n_final_markings_lst=None, 
                                   seed_1=101, seed_2=202, as_df=True, viz_proc_tree=False,
                                   return_alignment_res_dict=False, return_model=False,
                                   explor_reward=0.000, window_overlap=0, read_model_from_file=False, file_path='',
                                   activity_mapping_dict=None, use_trace_frequencies=False, return_agg_res_dict=False,
                                   reachability_heuristic=False, trace_recovery=False, map_activities_to_strings=True,
                                   include_duplicate_traces=True, non_sync_penalty=1,
                                   allow_train_test_intersection=True, cost_function=None, n_sampling_indices=5,
                                   round_precision=3, activity_prob_threshold=0, sequential_sampling=False, return_test_df=False,
                                   return_test_ground_truth=True, return_alignment_weights_dict=True,
                                   logarithmic_scaling_constant=4.6, hist_prob_dict=None):

    
    cost_res_dict = defaultdict(list)
    alignment_res_dict = defaultdict(list)
    time_res_dict = defaultdict(list)
    nodes_opened_dict = defaultdict(list)
    stochastic_acc_dict = defaultdict(list)
    argmax_acc_dict = defaultdict(list)
    alignment_weights_dict = defaultdict(list)
    
    if cost_function == 'logarithmic':
        cost_function = lambda x: -np.log(x) / logarithmic_scaling_constant
    
    # Default values
    if window_lengths_lst is None:
        window_lengths_lst = [3]
    if n_final_markings_lst is None:
        n_final_markings_lst = [1]

    result = train_test_log_split(
        df,
        n_train_traces,
        n_test_traces,
        train_traces,
        test_traces,
        random_trace_selection,
        seed_1,
        include_duplicate_traces,
        allow_train_test_intersection,
        sk_df,
        sftmax_lst
    )
    
    train_df = result.get('train_df')
    test_df = result.get('test_df')
    test_ground_truth = result.get('test_ground_truth')
    test_sftmax_lst = result.get('test_sftmax_lst')

    if test_sftmax_lst is not None:
        test_sftmax_lst = convert_tensors_to_numpy(test_sftmax_lst)
        stochastic_trace_df_lst = []
        test_ground_truth, test_sftmax_lst = select_random_indices_in_log_and_sftm_matrices_lst(test_df, 
                                                                                                test_sftmax_lst,
                                                                                                n_sampling_indices, 
                                                                                            sequential_sampling=sequential_sampling,
                                                                                                random_seed=seed_1) 
        
        test_traces_cases = list(test_ground_truth['case:concept:name'].unique())
        for idx, case in enumerate(test_traces_cases):

            stochastic_trace_df = sfmx_mat_to_sk_trace(test_sftmax_lst[idx], case,
                                                       round_precision=round_precision,
                                                       threshold=activity_prob_threshold) 
            
            stochastic_trace_df_lst.append(stochastic_trace_df)

        test_df = pd.concat(stochastic_trace_df_lst)  
    
    if read_model_from_file:
        if activity_mapping_dict is None:
            raise ValueError(f'To generate a process model from a file you must pass a mapping dictionary to be consistent with the log')
        net, init_marking, final_marking = generate_model_from_file(file_path, activity_mapping_dict=activity_mapping_dict)

    else:
        net, init_marking, final_marking, pm4py_model = prepare_process_model(train_df, non_sync_penalty=non_sync_penalty, cost_function=cost_function)

    
    if 'probs' not in test_df.columns:
        test_df['probs'] = [[1.0] for _ in range(len(test_df))]


    if reachability_heuristic:
        print(f'Started computing net_details_dict')
        net_details_dict = map_markings_to_reachable_transitions(net)
        print(f'Finished computing net_details_dict')
    else:
        net_details_dict = None


    for window_len in window_lengths_lst:

        for n_markings in n_final_markings_lst:

            print(f'Evaluating variant: n_markings={n_markings}, window_len={window_len}')

            for case in test_df['case:concept:name'].unique():
                trace_df = test_df[test_df['case:concept:name'] == case]
                start_time = time.time()

                dist, full_alignment, nodes_opened = horizon_based_conformance(net, trace_df,
                                                                 window_len=window_len,
                                                                 n_unique_final_markings=n_markings,
                                                                 explor_reward=explor_reward,
                                                                 window_overlap=window_overlap,
                                                                 net_details_dict=net_details_dict,
                                                                 reachability_heuristic=reachability_heuristic, 
                                                                 cost_function=cost_function,
                                                                 hist_prob_dict=hist_prob_dict)


                end_time = time.time()
                computation_time = end_time - start_time

                nodes_opened_dict[(n_markings, window_len)].append(nodes_opened)
                cost_res_dict[(n_markings, window_len)].append(dist)
                alignment_list = [t.name if isinstance(t, Transition) else t for t in full_alignment] + [dist]
                alignment_res_dict[(n_markings, window_len)].append(alignment_list)
                time_res_dict[(n_markings, window_len)].append(computation_time)  
                alignment_weights = [t.weight if isinstance(t, Transition) else t for t in full_alignment] + [dist]
                alignment_weights_dict[(n_markings, window_len)].append(alignment_weights)
                
                if test_ground_truth is not None:
                    clean_alignment = [t for t in full_alignment if isinstance(t, Transition)]
                    true_trace_df = test_ground_truth[test_ground_truth['case:concept:name'] == case]
                    argmax_accuracy = compute_argmax_accuracy(trace_df, true_trace_df)
                    stochastic_accuracy = compute_alignment_accuracy(clean_alignment, true_trace_df)
                    argmax_acc_dict[(n_markings, window_len)].append(argmax_accuracy)
                    stochastic_acc_dict[(n_markings, window_len)].append(stochastic_accuracy)

            
            if window_len is None:
                break
                
    
    if as_df:
        # Convert cost and time results into DataFrames
        res_df = defaultdict_to_dataframe(cost_res_dict, 'cost')
        time_df = defaultdict_to_dataframe(time_res_dict, 'time')
        nodes_opened_df = defaultdict_to_dataframe(nodes_opened_dict, 'nodes_opened')
        argmax_acc_df = defaultdict_to_dataframe(argmax_acc_dict, 'argmax_acc')
        stochastic_acc_df = defaultdict_to_dataframe(stochastic_acc_dict, 'stochastic_acc')
        
        # Merge the cost DataFrame and time DataFrame
        final_df = pd.concat([res_df, time_df, nodes_opened_df, stochastic_acc_df, argmax_acc_df], axis=1)

        for col in final_df.columns:
            if '_time' in col:
                final_df[col] = final_df[col].astype(float).round(3)  # Format time columns
            elif '_nodes_opened' in col:
                final_df[col] = final_df[col].astype(float).round(2)  # Format nodes opened columns
            else:
                final_df[col] = final_df[col].astype(float).round(2)  # Format cost columns

        # Assuming 'cases' list is predefined with the unique case IDs
        if 'case:concept:name' not in final_df.columns:
            final_df.insert(0, 'case:concept:name', test_df['case:concept:name'].unique().tolist())

        # Calculate and add relative time columns for multiple markings compared to one marking
        for window_len in window_lengths_lst:
            one_marking_time_col = f'marking=1, window_len={window_len}_time'

            for n_markings in n_final_markings_lst:
                if n_markings == 1:  # Skip the single marking case
                    continue

                multiple_markings_time_col = f'marking={n_markings}, window_len={window_len}_time'

        
        if use_trace_frequencies:
            trace_frequencies = create_case_id_to_frequency_dict(df)
            final_df['frequency'] = final_df['case:concept:name'].apply(lambda x: trace_frequencies.get(x, 0))

            aggregated_results = {}

            # Exclude the first column ('case:concept:name') from the aggregation process
            columns_to_aggregate = final_df.columns[1:]  # Skip the first column

            # Iterate over the columns to aggregate, calculating the weighted sum for each
            for column in columns_to_aggregate:
                # Check if the column is not the 'frequency' column; adjust the condition as needed
                if column != 'frequency':
                    # Calculate the weighted sum for the current column
                    weighted_sum = (final_df[column] * final_df['frequency']).sum()
                    # Store the result in the dictionary
                    aggregated_results[column] = weighted_sum


    return handle_return_logic(
        final_df,
        alignment_res_dict=alignment_res_dict,
        model=(net, init_marking, final_marking),
        return_alignment_res_dict=return_alignment_res_dict,
        return_model=return_model,
        return_test_df=return_test_df,
        test_df=test_df,
        return_test_ground_truth=return_test_ground_truth,
        test_ground_truth=test_ground_truth,
        alignment_weights_dict=alignment_weights_dict,
        return_alignment_weights_dict=return_alignment_weights_dict
    )


def handle_return_logic(
    final_df, 
    alignment_res_dict=None, 
    model=None, 
    return_alignment_res_dict=False, 
    return_model=False, 
    return_test_df=False, 
    test_df=None,
    return_test_ground_truth=False,
    test_ground_truth=None,
    alignment_weights_dict=None,
    return_alignment_weights_dict=False,
    add_ground_truth_to_test_df=True
):
    """
    Handles the return logic based on the provided flags.

    Args:
        final_df (pd.DataFrame): The final dataframe to return.
        alignment_res_dict (dict, optional): The alignment results dictionary.
        model (tuple, optional): The model tuple (net, init_marking, final_marking).
        return_alignment_res_dict (bool, optional): Whether to return the alignment results dictionary.
        return_model (bool, optional): Whether to return the model.
        return_test_df (bool, optional): Whether to return the test dataframe.
        test_df (pd.DataFrame, optional): The test dataframe to return.
        return_test_ground_truth (bool, optional): Whether to return the test ground truth dataframe.
        test_ground_truth (pd.DataFrame, optional): The test ground truth dataframe to return.

    Returns:
        tuple: A tuple containing the requested results based on the flags.
    """
    results = [final_df]

    if return_alignment_res_dict:
        results.append(alignment_res_dict)
        if return_model:
            results.append(model)
    elif return_model:
        results.append(model)

    if return_test_df:
        if add_ground_truth_to_test_df:
            test_df['ground_truth'] = test_ground_truth['concept:name']
        results.append(test_df)

    if return_test_ground_truth:
        results.append(test_ground_truth)

    if return_alignment_weights_dict:
        results.append(alignment_weights_dict)
    return tuple(results)


def defaultdict_to_dataframe(dd, suffix=''):
    """
    Convert a defaultdict to a pandas DataFrame with a customizable suffix.

    Parameters:
    dd (defaultdict): The defaultdict to convert.
    suffix (str): The suffix to append to the column names.

    Returns:
    pd.DataFrame: The resulting DataFrame.
    """
    # Convert defaultdict to dictionary
    dict_data = dict(dd)

    # Convert dictionary to DataFrame
    df = pd.DataFrame.from_dict(dict_data, orient='index')

    # Transpose DataFrame to get desired format
    df = df.transpose()
    
    # This assumes the keys in your defaultdict are tuples that you want to convert into a string format for the columns
    df.columns = ['marking={}, window_len={}_{}'.format(i, j, suffix) for i, j in df.columns]
    
    return df


def compare_one_vs_multiple_markings(file_path, n_markings=1, window_lengths=5, window_overlap=0,
                                     keep_only_unique_traces=False, return_freq_dict=False, seed_1=101, seed_2=202,
                                     train_traces=None, n_train_traces=None, test_traces=None, n_test_traces=None,
                                     use_trace_frequencies=False, return_agg_res_dict=True):  
    
    if not isinstance(file_path, pd.DataFrame):
        df = read_dataframe(file_path)
    else:    
        df = file_path   
        
    if return_freq_dict:
        freq_dict = trace_frequencies(df)
        
    if keep_only_unique_traces:
        df = get_unique_traces(df.copy())
        
    if not isinstance(n_markings, list):
        n_markings = [n_markings]
    
    if not isinstance(window_lengths, list):
        window_lengths = [window_lengths]
        
    res, alignments_dict, model_tuple, aggregated_results = compare_window_based_baselines(df=df,
                                                                       window_lengths_lst=window_lengths,
                                                                       n_final_markings_lst=n_markings,
                                                                       return_alignment_res_dict=True,
                                                                       return_model=True, seed_1=seed_1,
                                                                       seed_2=seed_2,
                                                                       explor_reward=0,
                                                                       train_traces=train_traces,
                                                                       n_train_traces=n_train_traces,
                                                                       test_traces=test_traces,
                                                                       n_test_traces=n_test_traces,
                                                                       window_overlap=window_overlap,
                                                                       use_trace_frequencies=use_trace_frequencies,
                                                                       return_agg_res_dict=return_agg_res_dict)
    
    
    if return_freq_dict:
        if return_agg_res_dict:
            return res, freq_dict, aggregated_results
        return res, freq_dict
    
    if return_agg_res_dict:
        return res, aggregated_results
    return res


def get_activity_multiset_single_trace(trace_df):
    """
    Constructs a multiset of activity labels for a single trace.
    
    Args:
    trace_df (pd.DataFrame): DataFrame containing the column 'concept:name' with activity labels.
    
    Returns:
    Counter: A Counter object representing the multiset of activity labels.
    """
    # Creating a multiset from the 'concept:name' column
    activity_multiset = Counter(trace_df['concept:name'])
    
    return activity_multiset


def subtract_activities(multiset, trace_df):
    """
    Subtracts the activities in trace_df from the given multiset and raises an error if any count goes negative.
    The function handles both a DataFrame of activities and a single activity string.

    Args:
    multiset (Counter): The original multiset of activities.
    trace_df (Union[pd.DataFrame, str]): DataFrame containing the column 'concept:name' with activity labels to subtract,
                                         or a single activity string to be subtracted.

    Returns:
    Counter: The updated multiset after subtraction.

    Raises:
    ValueError: If subtracting any activity results in a negative count.
    """
    if isinstance(multiset, dict) and not multiset:
        return multiset
    # Check if trace_df is a single string (single activity) and convert it to a Counter
    if isinstance(trace_df, str):
        trace_activities = Counter([trace_df])
    elif isinstance(trace_df, pd.DataFrame):
        trace_activities = Counter(trace_df['concept:name'])
    else:
        print(f'received the following type={type(trace_df)} which equals to {trace_df}')
        raise TypeError("trace_df must be either a pandas DataFrame or a string representing a single activity.")

    # Subtract activities
    for activity, count in trace_activities.items():
        if multiset[activity] < count or activity not in multiset:
            raise ValueError(f"Cannot subtract {count} instances of '{activity}' as it would result in a negative count.")
        multiset[activity] -= count
        if multiset[activity] == 0:
            del multiset[activity]

    return multiset  # Returning the updated multiset for clarity
  

# recursive version
# def map_markings_to_reachable_transitions(model):
#     """
#     Maps marking places to their reachable transitions in a Petri net model.

#     This function performs a breadth-first traversal of the Petri net's reachability graph,
#     starting from the initial marking. It tracks information about each marking place,
#     including its parent marking places and reachable transitions.

#     Args:
#         model: The Petri net model to analyze.

#     Returns:
#         A dictionary mapping markings to their associated details (parents, reachable transitions).
#     """

#     visited = set()
#     initial_marking_node = PetriNetNode(marking_places=model.init_mark.places)  # Removed depth
#     queue = deque([initial_marking_node])
#     marking_details = defaultdict(default_info)

#     while queue:
#         current_node = queue.popleft()
#         marking_places = current_node.marking_places

#         if marking_places in visited:
#             continue

#         visited.add(marking_places)

#         available_transitions = model._find_available_transitions(marking_places)
#         available_transitions_labels = {t.label for t in available_transitions if t.label}
#         # print(f'The available transitions are: {available_transitions_labels}')
#         update_reachable_transitions(marking_places, marking_details, available_transitions_labels)

#         for transition in available_transitions:
#             successor_marking = model._fire_transition(marking_places, transition)
#             successor_places = successor_marking.places
#             marking_details[successor_places]['parents'].add(marking_places)

#             if successor_places in visited:
#                 # Already visited, but update with any new reachable transitions from the current node.
#                 successor_marking_reachable_transitions = marking_details[successor_places]['reachable_transitions']
#                 update_reachable_transitions(marking_places, marking_details, successor_marking_reachable_transitions)
#             else:
#                 # Not visited yet, create a new node and add it to the queue.
#                 new_node = PetriNetNode(parent=current_node, transition_to_parent=transition, marking_places=successor_places)
#                 queue.append(new_node)

#     return marking_details

   
def update_multiset(multiset, activity):
    """
    Update the multiset by decrementing the count of the specified activity.
    If the activity's count reaches zero, it is removed from the multiset.
    Raises an exception if the activity does not exist in the multiset.

    Args:
    multiset (Counter): The multiset to update.
    activity (hashable): The element to decrement.

    Raises:
    ValueError: If the activity is not present in the multiset.
    """
    if multiset[activity] <= 0:
        raise ValueError(f'Activity {activity} does not appear inside the multiset')

    multiset[activity] -= 1  # Decrement the count of the element
    if multiset[activity] == 0:
        del multiset[activity]  # Remove the element if the count is zero
        

# def update_reachable_transitions(marking_places, marking_details, available_transitions_labels, visited=None):
#     """
#     Recursively updates reachable transitions for a set of marking places and their ancestors.

#     Args:
#         marking_places: A tuple representing a marking within the process model whose reachable transitions are to be updated.
#         marking_details: A dictionary containing marking details for each marking.
#         available_transitions_labels: The set of available transition labels.
#         visited: A set to track visited marking places (used internally to prevent infinite recursion).
#     """
    
#     if visited is None:  # Initialize the visited set if not provided
#         visited = set()
    
#     # Base case: No more predecessors to consider or already visited
#     if not marking_places or marking_places in visited:
#         return

#     visited.add(marking_places)  # Mark as visited to prevent cycles

#     try:
#         # Update reachable transitions
#         marking_details[marking_places]['reachable_transitions'].update(available_transitions_labels)
#     except KeyError:
#         # Handle cases where the marking place is not in the marking_details dictionary
#         print(f"Warning: Marking place '{marking_places}' not found in marking details.")
#         return
    
#     # Iterate over potential parent nodes based on marking details
#     for parent_marking_places in marking_details[marking_places]['parents']:
#         update_reachable_transitions(parent_marking_places, marking_details, available_transitions_labels, visited)    
        
    
def update_multiset(multiset, activity):
    """
    Update the multiset by decrementing the count of the specified activity.
    If the activity's count reaches zero, it is removed from the multiset.
    Raises an exception if the activity does not exist in the multiset.

    Args:
    multiset (Counter): The multiset to update.
    activity (hashable): The element to decrement.

    Raises:
    ValueError: If the activity is not present in the multiset.
    """
    if multiset[activity] <= 0:
        raise ValueError(f'Activity {activity} does not appear inside the multiset')

    multiset[activity] -= 1  # Decrement the count of the element
    if multiset[activity] == 0:
        del multiset[activity]  # Remove the element if the count is zero

        
def select_traces(log_df: pd.DataFrame, num_traces: int) -> pd.DataFrame:
    """
    Selects a specified number of traces from a log dataframe.

    Args:
        log_df (pd.DataFrame): The log dataframe with 'case:concept:name' as a column.
        num_traces (int): The number of unique traces to select.

    Returns:
        pd.DataFrame: A sublog containing the selected traces.

    Raises:
        ValueError: If `num_traces` exceeds the number of unique traces in the log.
    """
    
    unique_cases = log_df['case:concept:name'].unique()

    if num_traces > len(unique_cases):
        raise ValueError("Requested number of traces exceeds the number of unique traces in the log.")
    
    # Sample cases randomly, reset the index of the sampled cases to be sequential.
    selected_cases = pd.Series(unique_cases).sample(n=num_traces, replace=False).reset_index(drop=True)

    # Filter the log_df based on the selected cases
    return log_df[log_df['case:concept:name'].isin(selected_cases)]        


def select_random_indices_in_log_and_sftm_matrices_lst(
    log_df: pd.DataFrame, 
    sftm_mat_lst: list, 
    n_indices: int = 100,
    sequential_sampling: bool = False,
    random_seed: int = 42
):
    """
    Selects random indices in the log dataframe and corresponding softmax matrices list.
    Optionally, select indices from each sequential activity.

    Parameters:
    log_df (pd.DataFrame): DataFrame containing the log data with 'case:concept:name' column.
    sftm_mat_lst (list): List of softmax matrices corresponding to the traces in the log dataframe.
    n_indices (int, optional): Number of indices to select randomly. Default is 100.
    sequential_sampling (bool, optional): If True, sample indices from each sequential activity.
    random_seed (int, optional): Seed for random sampling. Default is 42.

    Returns:
    pd.DataFrame: Filtered log dataframe with selected indices.
    list: Filtered list of softmax matrices with selected indices.
    """
    # Set the random seed for reproducibility
    seed(random_seed)
    
    filtered_trace_lst = []
    filtered_sftm_mat_lst = []

    for i, case in enumerate(log_df['case:concept:name'].unique()):
        trace = log_df[log_df['case:concept:name'] == case]
        np_sftm_mat = sftm_mat_lst[i]
        trace.reset_index(drop=True, inplace=True)

        if len(trace) != np_sftm_mat.shape[1]:
            raise ValueError(f"Mismatch in lengths for case {case}: trace length is {len(trace)}, but matrix length is {np_sftm_mat.shape[1]}")

        if n_indices is None:
            n_indices = np_sftm_mat.shape[1]
    
        if sequential_sampling:
            selected_indices = []
            activity_start_pos = 0
        
            for i in range(1, len(trace)):
                if trace.iloc[i]['concept:name'] != trace.iloc[i - 1]['concept:name']:
                    # Sample from the previous group of consecutive activities
                    activity_positions = list(range(activity_start_pos, i))
                    n_sample = min(len(activity_positions), n_indices)
                    selected_indices += sample(activity_positions, n_sample)
                    activity_start_pos = i
        
            # Don't forget to sample from the last group
            activity_positions = list(range(activity_start_pos, len(trace)))
            n_sample = min(len(activity_positions), n_indices)
            selected_indices += sample(activity_positions, n_sample)
        
            selected_indices.sort()
        else:
            activity_positions = list(range(len(trace)))
            n_indices = min(n_indices, np_sftm_mat.shape[1])
            selected_indices = sorted(sample(activity_positions, n_indices))

        sftm_filtered = np_sftm_mat[:, selected_indices]
        trace_df_filtered = trace.iloc[selected_indices].reset_index(drop=True)

        filtered_trace_lst.append(trace_df_filtered)
        filtered_sftm_mat_lst.append(sftm_filtered)

    return pd.concat(filtered_trace_lst).reset_index(drop=True), filtered_sftm_mat_lst


def sfmx_mat_to_sk_trace(sftm_mat, case_num, round_precision=2, threshold=0.0):
    # If the input is a PyTorch tensor, convert it to a NumPy array
    if type(sftm_mat) is torch.Tensor:
        sftm_mat = sftm_mat.squeeze(0).cpu().numpy()

    df_prob_lst = []
    df_activities_lst = []

    for i in range(sftm_mat.shape[1]):
        probs = sftm_mat[:, i]
        activities = [str(act) for act in range(len(probs))]
        
        # Apply threshold filter
        filtered_probs = [prob for prob in probs if prob >= threshold]
        filtered_activities = [activities[j] for j in range(len(probs)) if probs[j] >= threshold]
        
        # Round the filtered probabilities
        rounded_probs = np.round(filtered_probs, round_precision)
        
        # Discard probabilities that round to 0
        final_probs = [prob for prob in rounded_probs if prob > 0]
        final_activities = [filtered_activities[j] for j in range(len(rounded_probs)) if rounded_probs[j] > 0]
        
        if not final_probs:  # Raise an exception if there are no valid probabilities
            raise ValueError(f"No valid probabilities found for index {i} after filtering and rounding.")
                
        df_prob_lst.append(final_probs)
        df_activities_lst.append(final_activities)

    case_lst = [case_num] * sftm_mat.shape[1]

    df = pd.DataFrame(
        {'case:concept:name': case_lst,
         'concept:name': df_activities_lst,
         'probs': df_prob_lst
        })

    return df


def map_to_string_numbers(
    df: pd.DataFrame, 
    map_dict: Dict[Union[int, str], str] = None, 
    return_map_dict: bool = True,
    map_strings_to_integer_strings: bool = False
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict[Union[int, str], str]]]:
    """
    Converts values in the 'concept:name' column of the DataFrame to their string equivalents.
    If `map_strings_to_integer_strings` is True, maps unique strings to sequential numbers starting from 0.

    Args:
        df (pd.DataFrame): The input DataFrame with a 'concept:name' column.
        map_dict (Dict[Union[int, str], str], optional): A dictionary for mapping values in 'concept:name' to strings. Defaults to None.
        return_map_dict (bool, optional): Whether to return the map_dict along with the DataFrame. Defaults to True.
        map_strings_to_integer_strings (bool, optional): Whether to map strings to sequential integer strings. Defaults to False.

    Returns:
        Union[pd.DataFrame, Tuple[pd.DataFrame, Dict[Union[int, str], str]]]: The modified DataFrame, and optionally the map_dict.

    Raises:
        ValueError: If a value in 'concept:name' cannot be converted to an integer.
    """
    def convert_to_string(value: Union[int, str]) -> str:
        try:
            return str(int(value))
        except ValueError:
            raise ValueError(f"Value '{value}' cannot be converted to an integer.")
    
    if map_strings_to_integer_strings:
        if map_dict is None:
            map_dict = {}
        current_max = max([int(v) for v in map_dict.values()]) + 1 if map_dict else 0

        def map_string_to_sequential_number(value: Union[int, str]) -> str:
            if value not in map_dict:
                nonlocal current_max
                map_dict[value] = str(current_max)
                current_max += 1
            return map_dict[value]
        
        df['concept:name'] = df['concept:name'].apply(map_string_to_sequential_number)
    else:
        if map_dict:
            df['concept:name'] = df['concept:name'].map(map_dict)
        else:
            df['concept:name'] = df['concept:name'].apply(convert_to_string)

    return (df, map_dict) if return_map_dict else df


def prepare_df(dataset_name: str, return_map_dict: bool = False, read_map_dict: bool = False, path: str = '') -> Union[pd.DataFrame, Tuple[pd.DataFrame, List, Dict], Tuple[pd.DataFrame, List]]:
    """
    Prepare a DataFrame from a specified dataset.

    Parameters:
    dataset_name (str): The name of the dataset ('50salads', 'gtea', or 'breakfast').
    return_map_dict (bool): Whether to return a dictionary mapping original values to string numbers.
    path (str): The path to the dataset files.

    Returns:
    Union[pd.DataFrame, Tuple[pd.DataFrame, List, Dict], Tuple[pd.DataFrame, List]]: A DataFrame containing the processed data, the
    softmax list, and optionally the mapping dictionary.
    """
    
    # Set default path if not provided
    if not path:
        path = os.path.join('C:\\Users\\User\\Jupyter Projects\\Research_2\\Datasets\\Video')
        
    # Define file names based on dataset name
    if dataset_name == '50salads':
        softmax_name = '50salads_softmax_lst.pickle'
        target_name = '50salads_target_lst.pickle'
    elif dataset_name == 'gtea':
        softmax_name = 'gtea_softmax_lst.pickle'
        target_name = 'gtea_target_lst.pickle'
    elif dataset_name == 'breakfast':
        softmax_name = 'breakfast_softmax_lst.pickle'
        target_name = 'breakfast_target_lst.pickle'
    else:
        raise ValueError("Dataset name must be '50salads', 'gtea', or 'breakfast'.")

    # Load data from pickle files
    with open(os.path.join(path, softmax_name), 'rb') as handle:
        softmax_lst = CPU_Unpickler(handle).load()

    with open(os.path.join(path, target_name), 'rb') as handle:
        target_lst = CPU_Unpickler(handle).load()

    # Prepare lists for DataFrame construction
    concat_tensor_lst = []
    concat_idx_lst = []

    for i, tensor in enumerate(target_lst):
        tensor_lst = tensor.tolist()
        tensor_lst = [str(elem) for elem in tensor_lst]
        idx_lst = [str(i)] * len(tensor_lst)
        concat_tensor_lst += tensor_lst
        concat_idx_lst += idx_lst

    # Create DataFrame
    df = pd.DataFrame(
        {
            'case:concept:name': concat_idx_lst,
            'concept:name': concat_tensor_lst
        }
    )
    
    if read_map_dict:
        mapping_dict = read_mapping_file(dataset_name)

    else:
        mapping_dict = None
        
    if return_map_dict:
        df, _ = map_to_string_numbers(df, mapping_dict)
        return df, softmax_lst, mapping_dict
    else:
        df, _ = map_to_string_numbers(df, mapping_dict)
        return df, softmax_lst


def read_mapping_file(dataset_name: str) -> dict:
    """
    Reads a mapping file for a given dataset and returns a dictionary mapping activities to string integers.

    Args:
        dataset_name (str): The name of the dataset (e.g., '50salads', 'gtea', 'breakfast').

    Returns:
        dict: A dictionary mapping activities to string integers.
    """
    # Construct the file path based on the dataset name
    file_path = f'C:\\Users\\User\\Jupyter Projects\\Research_2\\Datasets\\Video\\mapping_{dataset_name}.txt'

    # Initialize an empty dictionary
    activity_dict = {}

    # Open and read the file
    try:
        with open(file_path, 'r') as file:
            for line in file:
                # Split the line by whitespace to separate the integer and the activity
                parts = line.strip().split(maxsplit=1)
                if len(parts) == 2:
                    key, value = parts
                    activity_dict[value] = key
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

    return activity_dict


def compute_alignment_accuracy(alignment, true_trace_df: pd.DataFrame) -> float:
    """
    Computes the relative accuracy of the alignment by comparing it with the ground truth trace.

    Parameters:
    alignment (list): List of transitions with attributes 'label' and 'move_type'.
    true_trace_df (pd.DataFrame): DataFrame containing the ground truth trace with a 'case:concept:name' column.

    Returns:
    float: The relative accuracy of the alignment.
    """
    # Filter out transitions with label=None 
    filtered_alignment = [t for t in alignment if t.move_type in {'trace', 'sync'}]
    # print(f'The filteres alignment is: {filtered_alignment}')
    # Extract labels from the filtered alignment
    stochastic_pred = [t.label for t in filtered_alignment]
    # print(f'The stochastic pred is: {stochastic_pred}')
    # Extract ground truth values
    ground_truth = true_trace_df['concept:name'].tolist()
    
    # Check if the lengths of the lists are the same
    if len(stochastic_pred) != len(ground_truth):
        # print(f'Here are the stochastic predictions: {stochastic_pred} \n')
        # print(f'Here are the ground truth: {ground_truth}')
        raise ValueError("The lengths of stochastic predictions and ground truth do not match.")
    
    # Calculate the number of equal elements
    # print(f'The ground truth is {ground_truth}')
    num_equal_elements = sum(pred == truth for pred, truth in zip(stochastic_pred, ground_truth))
    
    # Calculate the relative equal occurrences
    stochastic_accuracy = num_equal_elements / len(ground_truth)
    
    return stochastic_accuracy


def compute_argmax_accuracy(stochastic_trace_df, true_trace_df):
    # Extract the highest probability prediction from probs column
    argmax_predictions = [
        concept_names[np.argmax(probs)]
        for concept_names, probs in zip(stochastic_trace_df['concept:name'], stochastic_trace_df['probs'])
    ]

    # Extract the ground truth values from true_trace_df
    ground_truth = true_trace_df['concept:name'].tolist()
    
    # Check if the lengths of the lists are the same
    if len(argmax_predictions) != len(ground_truth):
        # print(f'Here are the argmax predictions: {argmax_predictions} \n')
        # print(f'Here are the ground truth: {ground_truth}')
        raise ValueError("The lengths of stochastic predictions and ground truth do not match.")
    
    # Calculate the number of correct argmax predictions
    num_correct_argmax = sum(pred == truth for pred, truth in zip(argmax_predictions, ground_truth))
    
    # Calculate the relative argmax accuracy
    argmax_accuracy = num_correct_argmax / len(ground_truth)
    
    return argmax_accuracy


def compare_stochastic_vs_argmax_random_indices(
    df,
    softmax_lst=None,
    cost_function=None,
    n_train_traces=10,
    n_test_traces=10,
    train_cases=None,
    test_cases=None,
    n_indices=100,
    round_precision=2,
    random_trace_selection=True,
    random_seed=42,
    non_sync_penalty=1,
    activity_prob_threshold=0.0,
    fancy_cost_function=None,
    include_duplicate_traces=False,
    sequential_sampling=False,
    allow_intersection=True,
    only_return_model=False
):
    if cost_function == 'logarithmic':
        cost_function = lambda x: -np.log(x) / 4.7
    if cost_function == 'linear':
        cost_function = lambda x: 1 - x

    # Set random seed for reproducibility
    np.random.seed(random_seed)

    if softmax_lst is not None:
        softmax_lst = convert_tensors_to_numpy(softmax_lst)
        filtered_log_df, filtered_softmax_matrices = select_random_indices_in_log_and_sftm_matrices_lst(
            df, softmax_lst, n_indices, sequential_sampling=sequential_sampling
        )
    else:
        # Sample n_indices from each trace when softmax_lst is not provided
        filtered_log_df = sample_indices_from_df(df, n_indices, sequential_sampling)
        filtered_softmax_matrices = None

    # If train_cases or test_cases are provided, use them instead of random selection
    if train_cases is not None or test_cases is not None:
        random_trace_selection = False
        n_train_traces = len(train_cases) if train_cases is not None else n_train_traces
        n_test_traces = len(test_cases) if test_cases is not None else n_test_traces

    result = train_test_log_split(
        filtered_log_df, 
        n_train_traces=n_train_traces, 
        n_test_traces=n_test_traces,
        train_traces=train_cases,
        test_traces=test_cases,
        random_selection=random_trace_selection, 
        random_seed=random_seed,
        include_duplicate_traces=include_duplicate_traces, 
        allow_intersection=allow_intersection
    )
    
    train_df = result.get('train_df')
    test_df = result.get('test_df')
    
    train_df = prepare_df_cols_for_discovery(train_df)
    net, init_marking, final_marking = pm4py.discover_petri_net_inductive(train_df)
    model = from_discovered_model_to_PetriNet(net, non_sync_move_penalty=non_sync_penalty,
                                              pm4py_init_marking=init_marking, pm4py_final_marking=final_marking)
    
    if only_return_model:
        return model
           
    test_traces_cases = list(test_df['case:concept:name'].unique())
    alignment_lst_astar_extended = []
    alignment_lst_astar = []
    alignment_lst_dijkstra = []
    alignment_lst_reach = []
    
    for idx, trace_case in enumerate(test_traces_cases):
        true_trace_df = test_df[test_df['case:concept:name'] == trace_case].reset_index(drop=True)
        
        if filtered_softmax_matrices is not None:
            stmx_matrices_test = select_stmx_mats_for_test(filtered_softmax_matrices, test_df)
            stochastic_trace_df = sfmx_mat_to_sk_trace(
                stmx_matrices_test[idx], 
                trace_case,
                round_precision=round_precision,
                threshold=activity_prob_threshold
            )
        else:
            stochastic_trace_df = true_trace_df.copy()
            stochastic_trace_df['probs'] = [[1.0] for _ in range(len(stochastic_trace_df))]

        display(stochastic_trace_df)
        
        stochastic_trace = construct_stochastic_trace_model(stochastic_trace_df, non_sync_penalty)
        sync_prod = SyncProduct(model, stochastic_trace, cost_function=cost_function, fancy_cost_function=fancy_cost_function)
        
        # Uncomment the following line if you want to use Dijkstra's algorithm
        alignment_lst_dijkstra.append(sync_prod._dijkstra_no_rg_construct(trace_recovery=False)[0][0][1])
        # alignment_lst_astar.append(sync_prod.astar_search()[1])
        # alignment_lst_astar_extended.append(sync_prod.astar_incremental()[1])
        alignment_lst_reach.append(sync_prod.reach_search()[1])
    return alignment_lst_dijkstra, alignment_lst_reach


def construct_stochastic_trace_model(stochastic_trace_df: pd.DataFrame, non_sync_penalty: int = 1):
    """
    Constructs a stochastic trace model based on the given stochastic trace DataFrame.

    Parameters:
    stochastic_trace_df (pd.DataFrame): DataFrame containing 'concept:name' and 'probs' columns.
    non_sync_penalty (int, optional): Penalty for non-synchronous moves. Default is 1.

    Returns:
    stochastic_trace_model: Constructed stochastic trace model.
    """
    # Select relevant columns and reset index
    processed_df = stochastic_trace_df[['concept:name', 'probs']].reset_index(drop=True)
    
    # Construct the trace model
    stochastic_trace_model = construct_trace_model(processed_df, non_sync_penalty)
    
    return stochastic_trace_model
    

def sample_indices_from_df(df, n_indices, sequential_sampling):
    sampled_dfs = []
    for case in df['case:concept:name'].unique():
        trace_df = df[df['case:concept:name'] == case].reset_index(drop=True)
        sampled_trace = sample_from_trace(trace_df, n_indices, sequential_sampling)
        sampled_dfs.append(sampled_trace)
    
    return pd.concat(sampled_dfs, ignore_index=True)

def sample_from_trace(trace_df, n_indices, sequential_sampling):
    if sequential_sampling:
        # Group consecutive activities and sample from each group
        groups = trace_df.groupby((trace_df['concept:name'] != trace_df['concept:name'].shift()).cumsum())
        sampled_indices = []
        for _, group in groups:
            group_size = len(group)
            n_sample = min(n_indices, group_size)
            sampled_indices.extend(np.random.choice(group.index, n_sample, replace=False))
        return trace_df.loc[sorted(sampled_indices)]
    else:
        # Random sampling
        n_sample = min(n_indices, len(trace_df))
        return trace_df.sample(n=n_sample)

def prepare_df_cols_for_discovery(df):
    df_copy = df.copy()
    df_copy.loc[:, 'order'] = df_copy.groupby('case:concept:name').cumcount()
    
    if 'time:timestamp' in df_copy.columns:
        df_copy['time:timestamp'] = pd.to_datetime(df_copy['time:timestamp'])
    else:
        df_copy.loc[:, 'time:timestamp'] = pd.to_datetime(df_copy['order'])
    
    return df_copy


def print_cases_list(cases_list, list_name, max_elements=5):
    """
    Prints the beginning of a list if it's too long.

    Parameters:
    - cases_list (list): The list of cases to print.
    - list_name (str): The name of the list to display in the printout.
    - max_elements (int): The maximum number of elements to display before truncating.
    """
    if len(cases_list) > max_elements:
        displayed_cases = cases_list[:max_elements]
        print(f'The {list_name} are: {displayed_cases} ... [and {len(cases_list) - max_elements} more]\n')
    else:
        print(f'The {list_name} are: {cases_list} \n')


def train_test_log_split(
    dk_df: pd.DataFrame,
    n_train_traces: int = None,
    n_test_traces: int = None,
    train_traces: List[str] = None,
    test_traces: List[str] = None,
    random_selection: bool = True,
    random_seed: int = 42,
    include_duplicate_traces: bool = True,
    allow_intersection: bool = False,
    sk_df: pd.DataFrame = None,
    sftmax_lst: List[pd.DataFrame] = None
) -> Dict[str, Union[pd.DataFrame, List[pd.DataFrame]]]:
    """
    Splits the log into training and testing sets based on the specified number of unique traces or specific trace lists.

    Parameters:
    dk_df (pd.DataFrame): DataFrame containing the log data with 'case:concept:name' and 'concept:name' columns.
    n_train_traces (int, optional): Number of traces to include in the training set. Default is None.
    n_test_traces (int, optional): Number of traces to include in the testing set. Default is None.
    train_traces (List[str], optional): List of specific trace IDs to include in the training set. Default is None.
    test_traces (List[str], optional): List of specific trace IDs to include in the testing set. Default is None.
    random_selection (bool, optional): Whether to randomly select the traces. Default is True.
    random_seed (int, optional): Seed for random selection. Default is 42.
    include_duplicate_traces (bool, optional): Whether to allow multiple traces with identical sequences. Default is True.
    allow_intersection (bool, optional): Whether to allow the same trace sequences in both training and testing sets. Default is False.
    sk_df (pd.DataFrame, optional): Second DataFrame to use as the source for the test set. Default is None.
    sftmax_lst (List[pd.DataFrame], optional): List of softmax matrices corresponding to the traces in dk_df. Default is None.

    Returns:
    Dict[str, Union[pd.DataFrame, List[pd.DataFrame]]]: A dictionary with keys 'train_df', 'test_df', 'test_ground_truth',
    and 'test_sftmax_lst'.
    """
    
    result = {}
    cases_list = get_unique_cases(dk_df, include_duplicate_traces=include_duplicate_traces)

    if train_traces is not None:
        train_cases = train_traces
        if test_traces is not None:
            test_cases = test_traces
        else:
            remaining_cases = cases_list if allow_intersection else [case for case in cases_list if case not in train_traces]
            test_cases = select_cases(remaining_cases, n_test_traces, random_selection, random_seed)
    elif test_traces is not None:
        test_cases = test_traces
        remaining_cases = cases_list if allow_intersection else [case for case in cases_list if case not in test_traces]
        train_cases = select_cases(remaining_cases, n_train_traces, random_selection, random_seed)
    else:
        train_cases, test_cases = split_cases(cases_list, n_train_traces, n_test_traces, allow_intersection, random_selection, random_seed)

    print_cases_list(train_cases, 'train cases')
    print_cases_list(test_cases, 'test cases')
    
    if sk_df is not None:
        result['train_df'] = filter_dataframe(dk_df, train_cases)
        result['test_df'] = filter_dataframe(sk_df, test_cases)
        result['test_ground_truth'] = filter_dataframe(dk_df, test_cases)
        return result

    if sftmax_lst is not None:
        if len(dk_df['case:concept:name'].unique()) != len(sftmax_lst):
            raise ValueError("The length of dk_df and sftmax_lst must be the same")
        
        result['train_df'] = filter_dataframe(dk_df, train_cases)
        result['test_df'] = filter_dataframe(dk_df, test_cases)
        
        # Map the test cases to their original indices
        original_case_indices = {case: i for i, case in enumerate(dk_df['case:concept:name'].unique())}
        test_indices = [original_case_indices[case] for case in test_cases]
        result['test_sftmax_lst'] = [sftmax_lst[i] for i in test_indices]
        
        return result

    result['train_df'] = filter_dataframe(dk_df, train_cases)
    result['test_df'] = filter_dataframe(dk_df, test_cases)
    
    return result

def split_cases(
    cases_list: List[str], 
    n_train: int = None, 
    n_test: int = None, 
    allow_intersection: bool = True, 
    random_selection: bool = True, 
    random_seed: int = 42
) -> Tuple[List[str], List[str]]:
    
    if random_selection:
        random.seed(random_seed)
        random.shuffle(cases_list)

    if n_train is None and n_test is None:
        n_train = len(cases_list) // 2
        n_test = len(cases_list) - n_train
    elif n_train is None:
        n_train = len(cases_list) - n_test
    elif n_test is None:
        n_test = len(cases_list) - n_train

    if not allow_intersection and n_train + n_test > len(cases_list):
        raise ValueError("More training and testing traces were demanded than are available without intersection")

    if allow_intersection:
        train_cases = cases_list[:n_train]
        test_cases = cases_list[:n_test]
    else:
        train_cases = cases_list[:n_train]
        test_cases = cases_list[n_train:n_train + n_test]

    return train_cases, test_cases

def select_cases(
    cases_list: List[str],
    n_cases: int = None,
    random_selection: bool = True,
    random_seed: int = 42
) -> List[str]:
    """
    Selects a specified number of cases from a list of cases.

    Parameters:
    cases_list (List[str]): List of all unique case IDs.
    n_cases (int, optional): Number of cases to select. Default is None, which means all cases will be selected.
    random_selection (bool, optional): Whether to randomly select the cases. Default is True.
    random_seed (int, optional): Seed for random selection. Default is 42.

    Returns:
    List[str]: A list of selected case IDs.
    """
    if random_selection:
        random.seed(random_seed)
        random.shuffle(cases_list)
    
    if n_cases is None or n_cases > len(cases_list):
        return cases_list
    else:
        return cases_list[:n_cases]


def filter_dataframe(df: pd.DataFrame, cases: List[str]) -> pd.DataFrame:
    # Filter the dataframe to include only the specified cases
    filtered_df = df[df['case:concept:name'].isin(cases)]
    # Set the 'case:concept:name' column as the index
    filtered_df = filtered_df.set_index('case:concept:name')
    # Reindex according to the order of cases and reset the index
    ordered_df = filtered_df.loc[cases].reset_index()
    return ordered_df
    

def create_cost_function(epsilon=1, tao_weight=0.0000001):
    def cost_function_paper(transition):
        if transition.move_type not in {'sync', 'model', 'trace'}:  
            raise ValueError(f"Transition {transition} has a move type = {transition.move_type} which is not correct")
        elif transition.label is None:
            return tao_weight
        elif transition.move_type == 'sync':
            return -math.log(transition.prob)
        elif transition.move_type == 'trace':
            return -math.log(transition.prob) - math.log(epsilon)
        else:
            return -math.log(epsilon)
    return cost_function_paper


def convert_tensors_to_numpy(softmax_lst):
    numpy_lst = []
    for item in softmax_lst:
        if torch.is_tensor(item):
            numpy_item = item.numpy().copy()  # Convert to numpy array and make a copy
            numpy_item = np.squeeze(numpy_item, axis=0)  # Remove the first dimension using squeeze
            numpy_lst.append(numpy_item)
        else:
            numpy_lst.append(item)
    return numpy_lst


def alignment_dict_to_dataframe(alignment_dict: Dict[Tuple[int, int], List[List[str]]], 
                                alignment_weights_dict: Dict[Tuple[int, int], List[List[float]]] = None,
                                cases: List[int] = None) -> pd.DataFrame:
    """
    Converts an alignment dictionary to a DataFrame with three columns: marking_window_comb, case, and alignment.
    If alignment_weights_dict is provided, adds a new column with the move weights.

    Args:
        alignment_dict (Dict[Tuple[int, int], List[List[str]]]): The alignment dictionary.
        alignment_weights_dict (Dict[Tuple[int, int], List[List[float]]], optional): A dictionary with the same keys as alignment_dict
            containing lists of weights for each move.
        cases (List[int], optional): A list of unique case IDs. If provided, these case IDs will be used instead of sequential numbering.

    Returns:
        pd.DataFrame: A DataFrame with the specified columns.
    """
    data = []
    case_counter = 0
    
    for marking_window_comb, lists in alignment_dict.items():
        weights_lists = alignment_weights_dict.get(marking_window_comb, [[]] * len(lists)) if alignment_weights_dict else [[]] * len(lists)
        
        for lst, weights_lst in zip(lists, weights_lists):
            case_id = cases[case_counter] if cases else case_counter
            for i, item in enumerate(lst):
                row = {'marking_window_comb': marking_window_comb, 'case': case_id, 'alignment': item}
                if alignment_weights_dict:
                    row['weight'] = weights_lst[i] if i < len(weights_lst) else None
                data.append(row)
            case_counter += 1
    
    df = pd.DataFrame(data)
    return df


def group_similar_traces(df: pd.DataFrame) -> pd.DataFrame:
    # Group by 'case:concept:name' and aggregate the 'concept:name' into a tuple
    grouped = df.groupby('case:concept:name')['concept:name'].apply(tuple).reset_index()
    
    # Group by the trace (sequence of activities) and aggregate the case IDs
    trace_groups = grouped.groupby('concept:name')['case:concept:name'].apply(list).reset_index()
    
    # Add a column for the length of each trace
    trace_groups['trace_length'] = trace_groups['concept:name'].apply(len)
    
    # Sort case_list numerically
    trace_groups['case:concept:name'] = trace_groups['case:concept:name'].apply(lambda x: sorted(x, key=int))
    
    # Rename columns for clarity
    trace_groups.columns = ['trace', 'case_list', 'trace_length']
    
    return trace_groups[['case_list', 'trace_length']]


def sample_cases(trace_groups: pd.DataFrame, n: int = 1) -> list:
    # Function to sample n cases from each case_list
    def sample_case_list(case_list, n):
        return random.sample(case_list, min(len(case_list), n))
    
    # Apply the sampling function to each row and collect the results
    sampled_cases = []
    for case_list in trace_groups['case_list']:
        sampled_cases.extend(sample_case_list(case_list, n))
    
    # Sort the sampled cases numerically
    sampled_cases = sorted(sampled_cases, key=int)
    
    return sampled_cases


def filter_df_by_cases(df: pd.DataFrame, sampled_cases: list) -> pd.DataFrame:
    # Filter the DataFrame to only include rows with 'case:concept:name' in sampled_cases
    filtered_df = df[df['case:concept:name'].isin(sampled_cases)]
    return filtered_df


def filter_softmax_matrices(softmax_lst: list, sampled_cases: list) -> list:
    # Convert sampled_cases to integers to use as indices
    indices = [int(case) for case in sampled_cases]
    
    # Filter the softmax matrices based on the indices
    filtered_softmax_lst = [softmax_lst[i] for i in indices]
    
    return filtered_softmax_lst


def build_conditioned_prob_dict(df_train, max_hist_len=2, precision=2):

    def get_histories_up_to_length_k(activities_seq_list, k):
        """
        Generate all possible histories up to length k from the activities sequence.
        """
        histories = []
        for i in range(1, len(activities_seq_list)):
            for j in range(1, min(i, k) + 1):
                history = tuple(activities_seq_list[i-j:i+1])
                histories.append(history)
        return histories

    def get_relative_freq_dict(counter, precision=2):
        """
        Convert absolute counts to relative frequencies.
        """
        frequencies_dict = dict(Counter(counter))
        rel_freq_dict = defaultdict(dict)
        
        for key, freq in frequencies_dict.items():
            if len(key) > 1:
                prefix = key[:-1]
                total_prefix_freq = sum([frequencies_dict[sub_key] for sub_key in frequencies_dict if sub_key[:-1] == prefix])
                probability = round(freq / total_prefix_freq, precision)
                if probability > 0:
                    rel_freq_dict[prefix][key[-1]] = probability
        
        # Remove any empty dictionaries that may have resulted from filtering
        rel_freq_dict = {k: v for k, v in rel_freq_dict.items() if v}
                
        return rel_freq_dict
    
    # Convert 'concept:name' to string type if not already
    df_train['concept:name'] = df_train['concept:name'].astype(str)
    
    # Concatenate activities for each case, using a delimiter
    case_activities = df_train.groupby('case:concept:name')['concept:name'].apply(list).tolist()
    
    # Build the counter for histories
    counter = Counter()
    for activities_seq_list in case_activities:
        counter.update(get_histories_up_to_length_k(activities_seq_list, k=max_hist_len))
    
    # Compute relative frequencies
    rel_freq_dict = get_relative_freq_dict(counter, precision=precision)
    
    return dict(rel_freq_dict)