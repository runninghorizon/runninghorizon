from RunningHorizon import *
from Classes import *
from utils import *
import pm4py
from pm4py.objects.petri_net.utils import petri_utils
from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.objects.petri_net.semantics import enabled_transitions, execute
from pm4py.visualization.petri_net import visualizer as pn_visualizer
from pm4py.visualization.process_tree import visualizer as pt_visualizer
from pm4py.objects.conversion.wf_net.converter import apply as pt_converter
from pm4py.objects.process_tree.obj import ProcessTree, Operator
from pm4py.objects.process_tree.utils import generic as pt_util
from pm4py.objects.petri_net.utils.reachability_graph import construct_reachability_graph as crg
import numpy as np


# Strip process tree of XOR nodes
def remove_xor_nodes(node):
    if node.operator == Operator.XOR:
        return None  # Remove this node and its subtree
    
    filtered_children = []
    for child in node.children:
        result = remove_xor_nodes(child)
        if result is not None:
            filtered_children.append(result)
    
    node.children = filtered_children
    return node


# Get the names of transitions that were left in the process tree after the stripping
def extract_mandatory_nodes(node):
    """
    Extracts the mandatory nodes from a given process tree node, ensuring no None values are included in the result.

    Args:
        node (ProcessTree): The root node of the process tree.

    Returns:
        list: A list of mandatory node labels, excluding any None values.
    """
    mandatory_nodes = []
    
    if node.operator not in [Operator.XOR]:
        if isinstance(node, ProcessTree) and node.label is not None:
            mandatory_nodes.append(node.label)
        
        for child in node.children:
            mandatory_nodes.extend(extract_mandatory_nodes(child))
    
    # Remove None values
    mandatory_nodes = [mn for mn in mandatory_nodes if mn is not None]

    return mandatory_nodes


# Get all the reachable transitions from a net with the given initial marking
def get_reachable_transitions(net, initial_marking):
    # Form the reachability graph
    reachablity_graph = crg(net, initial_marking)

    # Create a new empty set for the reachable transitions
    reachable_transitions = set()

    # Extract all the reachable transitions from the reachability graph
    # Each transition here is of a pm4py...TransitionSystem.Transition object type, not the regular pm4py Transition.
    # This object doesn't have a label like the regular one, but a 'name' that is a string that needs to be stripped...
    for transition in reachablity_graph.transitions:
        if not transition.name.split(', ')[1].startswith("None"):
            reachable_transitions.add(transition.name.split(', ')[1][1:-2])

    return reachable_transitions
    

# Create a subnet of the original net for the given initial and final markings using the reachable transitions only
def create_subnet(net, reachable_transitions, initial_marking, final_marking):
    flag = True
    places = set()
    arcs = set()
    transitions = set()

    # Step 1: Add reachable transitions to the transitions set
    for transition in net.transitions:
        if transition.label in reachable_transitions:
            transitions.add(transition)

    # Step 2: Add places from initial and final markings to the places set
    for place in initial_marking:
        places.add(place)
    for place in final_marking:
        places.add(place)

    # Step 3: Add arcs connected to the reachable transitions and the places they're connected to to the corresponding sets
    for arc in net.arcs:
        # Arc from a place to a transition
        if isinstance(arc.source, PetriNet.Place) and isinstance(arc.target, PetriNet.Transition):
            if arc.target in transitions:
                places.add(arc.source)
                arcs.add(arc)

        # Arc from a transition to a place
        elif isinstance(arc.source, PetriNet.Transition) and isinstance(arc.target, PetriNet.Place):
            if arc.source in transitions:
                places.add(arc.target)
                arcs.add(arc)

    # Step 4: Iteratively add remaining reachable places and transitions until no new ones are found
    while flag:
        flag = False
        for arc in net.arcs:
            if arc not in arcs:
                # Every arc starting in a place that is already in the places set is added.
                # The transition it reaches is added as well.
                if isinstance(arc.source, PetriNet.Place) and arc.source in places:
                    arcs.add(arc)
                    transitions.add(arc.target)
                    flag = True

                # Every arc going out of a transition already in the transitions set is added.
                # The place it reaches is added as well.
                elif isinstance(arc.source, PetriNet.Transition) and arc.source in transitions:
                    arcs.add(arc)
                    places.add(arc.target)
                    flag = True

    # Step 5: Create the new subnet with identified places, transitions, and arcs
    sub_net = PetriNet(name="SubNet")
    
    for place in places:
        sub_net.places.add(place)
    
    for transition in transitions:
        sub_net.transitions.add(transition)
    
    for arc in arcs:
        sub_net.arcs.add(arc)

    # Step 6: Handle initial marking with more than one place
    if len(initial_marking) > 1:
        dummy_place = PetriNet.Place("dummy_place")
        tau_transition = PetriNet.Transition("tau_transition", None)
        
        sub_net.places.add(dummy_place)
        sub_net.transitions.add(tau_transition)
        
        # Connect dummy place to tau transition
        petri_utils.add_arc_from_to(dummy_place, tau_transition, sub_net)
        
        # Connect tau transition to the original initial marking places
        for place in initial_marking:
            petri_utils.add_arc_from_to(tau_transition, place, sub_net)
        
        # Create new initial marking with a token in the dummy place
        new_initial_marking = Marking()
        new_initial_marking[dummy_place] = 1
    else:
        new_initial_marking = initial_marking

    return sub_net, new_initial_marking


# The main function of the second heuristic
def get_mandatory_transitions_for_marking(net, initial_marking, final_marking):
    """
    Get a list of mandatory transitions for a given marking in a Petri net.

    Args:
        net (PetriNet): The Petri net.
        initial_marking (Marking): The initial marking.
        final_marking (Marking): The final marking.

    Returns:
        list: A list of mandatory transitions.
    """
    # Get a list of reachable transitions from the current place
    reachable_transitions = get_reachable_transitions(net, initial_marking)
    old = initial_marking
    
    # Create a subnet of the original net with only reachable transitions and places
    sub_net, initial_marking = create_subnet(net, reachable_transitions, initial_marking, final_marking)

    # Convert the subnet into a process tree
    try:
        process_tree = pt_converter(sub_net, initial_marking, final_marking)
    except ValueError as e:
        print("Error during conversion to process tree:", e)
        
        # Debugging information
        # Visualize the subnet before returning
        gviz = pn_visualizer.apply(sub_net, initial_marking, final_marking)
        pn_visualizer.view(gviz)
        raise

    # Remove the XOR nodes from the tree
    filtered_process_tree = remove_xor_nodes(process_tree)

    # We didn't get an empty tree or something after filtering it
    if filtered_process_tree:
        # Extract a list of all transitions left - the mandatory ones.
        mandatory_transitions = extract_mandatory_nodes(filtered_process_tree)
        return mandatory_transitions

    # Filtered process tree was empty - no mandatory transitions
    else:
        return []


def map_markings_to_mandatory_transitions(net, initial_marking, final_marking):
    markings_dict = {}
    visited = []
    queue = [initial_marking]

    while queue:
        current_marking = queue.pop(0)
        visited.append(current_marking)
        
        if current_marking in markings_dict:
            continue
        
        if current_marking == final_marking:
            markings_dict[current_marking] = set()
            continue
        
        mandatory_transitions = set(get_mandatory_transitions_for_marking(net, current_marking, final_marking))
        
        markings_dict[current_marking] = mandatory_transitions

        available_transitions = enabled_transitions(net, current_marking)
        for transition in available_transitions:
            new_marking = execute(transition, net, current_marking)
            if new_marking not in visited and new_marking not in queue:
                queue.append(new_marking)

    return markings_dict


def compute_mandatory_transitions(net, init_marking, final_marking):
    original_result_dict = map_markings_to_mandatory_transitions(net, init_marking, final_marking)
    
    petri_net_model, places_mapping = from_discovered_model_to_PetriNet(net, return_mapping=True)

    new_result_dict = {}
    for key, value in original_result_dict.items():
        new_marking = convert_marking(key, places_mapping)
        new_result_dict[new_marking] = value
    
    return petri_net_model, new_result_dict