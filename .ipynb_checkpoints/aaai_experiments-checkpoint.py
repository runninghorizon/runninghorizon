import pandas as pd  
import numpy as np  
import random 
import time  
from typing import List, Dict, Tuple  
from RunningHorizon import *
import time
from threading import Thread
import pickle

def filter_log_efficiently(log: pd.DataFrame, 
               min_len: int = None, 
               max_len: int = None, 
               n_traces: int = None,
               random_seed: int = None) -> pd.DataFrame:
    """
    Filters the log based on trace length (min_len and max_len) and the number of traces.

    Parameters:
    - log (pd.DataFrame): The event log as a pandas DataFrame.
    - min_len (int, optional): Minimum allowed length of traces. No lower limit if None.
    - max_len (int, optional): Maximum allowed length of traces. No upper limit if None.
    - n_traces (int, optional): Number of traces to include. Includes all if None.
    - random_seed (int, optional): Seed for reproducible random trace selection. No randomness if None.

    Returns:
    - pd.DataFrame: The filtered log as a pandas DataFrame.
    """

    # Filter based on min_len and max_len
    if min_len is not None or max_len is not None:
        trace_lengths = log.groupby('case:concept:name').size().reset_index(name='length')
        
        if min_len is not None:
            trace_lengths = trace_lengths[trace_lengths['length'] >= min_len]
        
        if max_len is not None:
            trace_lengths = trace_lengths[trace_lengths['length'] <= max_len]
        
        accepted_cases = trace_lengths['case:concept:name'].tolist()
    else:
        accepted_cases = log['case:concept:name'].unique()
    
    # If n_traces is None, return the filtered log
    if n_traces is None:
        return log[log['case:concept:name'].isin(accepted_cases)]
    
    # If n_traces is specified, limit the number of cases with reproducibility
    if random_seed is not None:
        selected_cases = pd.Series(accepted_cases).sample(n=min(len(accepted_cases), n_traces), random_state=random_seed).tolist()
    else:
        selected_cases = accepted_cases[:n_traces]
    
    return log[log['case:concept:name'].isin(selected_cases)]


def compare_search_algorithms(
    df: pd.DataFrame,
    df_name: str,
    cost_function: Optional[Any] = None,  # Replace 'Any' with the actual type if known
    n_train_traces: int = 10,
    n_test_traces: int = 10,
    train_cases: Optional[List[str]] = None,
    test_cases: Optional[List[str]] = None,
    random_seed: int = 304,
    non_sync_penalty: int = 1,
    only_return_model: bool = False,
    allow_intersection: bool = False,
    read_model_from_file: bool = False,
    model_path: Optional[str] = None,
    activity_mapping_dict: Optional[Dict[str, str]] = None,
    algorithms: Optional[List[str]] = None,
    time_limit: Optional[int] = None,
    return_results: bool = False  # New parameter to control whether to return or save the results
) -> Optional[Any]:  # Replace 'Any' with the actual return type of prepare_model if known
    """
    Function to compare specified search algorithms (A*, A* Extended, Reach) 
    by either training a model on the training data or reading it from a file, 
    and then evaluating it on the test data. Optionally returns the results 
    instead of saving them.
    """
    if algorithms is None:
        algorithms = ['astar', 'astar_extended', 'reach']

    cost_function = select_cost_function(cost_function)
    np.random.seed(random_seed)
    
    result = train_test_log_split(
        df, 
        n_train_traces=n_train_traces, 
        n_test_traces=n_test_traces,
        train_traces=train_cases,
        test_traces=test_cases,
        random_selection=(train_cases is None and test_cases is None), 
        random_seed=random_seed,
        allow_intersection=allow_intersection
    )
    
    train_df, test_df = result['train_df'], result['test_df']
    
    if read_model_from_file:
        if not model_path or not activity_mapping_dict:
            raise ValueError("Both 'model_path' and 'activity_mapping_dict' must be provided when reading a model from a file.")
        
        model = generate_model_from_file(
            model_path,
            activity_mapping_dict=activity_mapping_dict,
            return_markings=False
        )
    else:
        model = prepare_model(train_df, non_sync_penalty)
    
    if only_return_model:
        return model

    model = add_transition_mappings_to_model(model)
 
    # Group similar traces
    trace_dict, lookup_dict = group_similar_traces(test_df)
    total_traces = len(trace_dict)
    
    # Initialize results dictionary
    results = {alg: {'cost': [], 'time': [], 'nodes_popped': [], 'case_id': []} for alg in algorithms}
    
    computed_results = {}
    
    print(f"Overall computing statistics for {len(trace_dict)} traces")
    for idx, (trace, cases) in enumerate(trace_dict.items(), 1):
        print(f'\rComputing trace {idx}/{total_traces}', end='')
        representative_case = cases[0]
        true_trace_df = test_df[test_df['case:concept:name'] == representative_case].reset_index(drop=True)
        true_trace_df['probs'] = [[1.0] for _ in range(len(true_trace_df))]
        stochastic_trace = construct_stochastic_trace_model(true_trace_df, non_sync_penalty)
        sync_prod = SyncProduct(model, stochastic_trace, cost_function=cost_function)
        
        computed_results[trace] = {}
        
        for alg in algorithms:
            computed_results[trace][alg] = run_algorithm_with_timeout(
                sync_prod, alg, timeout=time_limit
            )

        # Store results for all cases with the same trace
        for case_id in cases:
            for alg in algorithms:
                if computed_results[trace][alg]:
                    results[alg]['cost'].append(computed_results[trace][alg]['cost'])
                    results[alg]['time'].append(time_limit)  # Store the time limit as the time taken
                    results[alg]['nodes_popped'].append(computed_results[trace][alg]['nodes_popped'])
                else:
                    results[alg]['cost'].append(None)
                    results[alg]['time'].append(None)
                    results[alg]['nodes_popped'].append(None)
                results[alg]['case_id'].append(case_id)

    if return_results:
        return results

    # Save results as DataFrames with the dataframe name included in the filename
    for alg in algorithms:
        save_results_as_dataframe(results[alg], df_name, alg)

    return None  # Return None if only_return_model is False and no other return is specified
    

def group_similar_traces(df):
    # Group by case:concept:name and aggregate activities into a tuple
    grouped_traces = df.groupby('case:concept:name')['concept:name'].apply(tuple)
    
    # Create a dictionary to store the groups of similar traces
    trace_dict = defaultdict(list)
    
    # Populate the dictionary
    for case_id, trace in grouped_traces.items():
        trace_dict[trace].append(case_id)
    
    # Create a lookup dictionary to find similar traces
    lookup_dict = {}
    for trace, case_ids in trace_dict.items():
        for case_id in case_ids:
            lookup_dict[case_id] = [cid for cid in case_ids if cid != case_id]
    
    return trace_dict, lookup_dict


def select_cost_function(cost_function):
    if cost_function == 'logarithmic':
        return lambda x: -np.log(x) / 4.7
    elif cost_function == 'linear':
        return lambda x: 1 - x
    return cost_function


def prepare_model(train_df: pd.DataFrame, non_sync_penalty: int):
    """
    Prepares a PetriNet model by discovering it from the provided data and computing the necessary transition mappings.

    Parameters:
    - train_df (pd.DataFrame): The training data as a pandas DataFrame.
    - non_sync_penalty (int): Penalty value for non-synchronous moves.

    Returns:
    - model: The prepared PetriNet model.
    """
    train_df = prepare_df_cols_for_discovery(train_df)
    net, init_marking, final_marking = pm4py.discover_petri_net_inductive(train_df)
    model = from_discovered_model_to_PetriNet(
        net, 
        non_sync_move_penalty=non_sync_penalty,
        pm4py_init_marking=init_marking, 
        pm4py_final_marking=final_marking
    )

    return model


def add_transition_mappings_to_model(model):
    """
    Computes the mandatory and alive transitions for the given PetriNet model.

    Parameters:
    - model: The PetriNet model object.

    Returns:
    - model: The updated PetriNet model with mandatory and alive transitions computed.
    """
    # Measure time for computing mandatory transitions
    if model.mandatory_transitions_map is None:
        print('Starting to compute mandatory transitions...')
        start_time = time.time()
        
        model.mandatory_transitions_map = compute_mandatory_transitions(
            model.pm4py_net, 
            model.pm4py_initial_marking, 
            model.pm4py_final_marking,
            model.place_mapping
        )
        
        end_time = time.time()
        print(f"Mandatory transitions computed in {end_time - start_time:.4f} seconds")

    # Measure time for computing alive transitions
    if model.alive_transitions_map is None:
        print('Starting to compute alive transitions...')
        start_time = time.time()
        
        model.alive_transitions_map = map_markings_to_reachable_transitions(model)
        
        end_time = time.time()
        print(f"Alive transitions computed in {end_time - start_time:.4f} seconds")
    
    return model


def run_search_algorithm(sync_prod, algorithm: str, results: Dict[str, Dict[str, List]], trace_case: str, store_result: bool = True):
    start_time = time.time()
    
    if algorithm == 'astar':
        alignment, cost, nodes_popped = sync_prod.astar_search()
    elif algorithm == 'astar_extended':
        alignment, cost, nodes_popped = sync_prod.astar_incremental()
    elif algorithm == 'reach':
        alignment, cost, nodes_popped = sync_prod.reach_search()
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    elapsed_time = time.time() - start_time
    
    if store_result:
        results[algorithm]['cost'].append(cost)
        results[algorithm]['time'].append(elapsed_time)
        results[algorithm]['nodes_popped'].append(nodes_popped)
        results[algorithm]['case_id'].append(trace_case)
    
    # Return the results so they can be reused if needed
    return {'cost': cost, 'time': elapsed_time, 'nodes_popped': nodes_popped}

def save_results_as_dataframe(results: Dict[str, List], df_name: str, algorithm_name: str):
    """
    Save the results as a DataFrame to a CSV file with the dataframe name included in the filename.

    Parameters:
    - results (Dict[str, List]): The results dictionary to be saved.
    - df_name (str): The name of the dataframe, used in the filename.
    - algorithm_name (str): The name of the algorithm, used in the filename.
    """
    # Convert the results dictionary into a DataFrame
    df = pd.DataFrame(results)
    
    # Ensure 'case_id' is the first column
    df = df[['case_id'] + [col for col in df.columns if col != 'case_id']]
    
    # Round 'cost' and 'time' columns to 5 decimal places
    df['cost'] = df['cost'].round(5)
    df['time'] = df['time'].round(5)
    
    # Construct the filename with the dataframe name and algorithm name
    filename = f"{df_name}_{algorithm_name}_results.csv"
    
    # Save the DataFrame to a CSV file
    df.to_csv(filename, index=False)


def load_and_preprocess_log(df_name: str, 
                            min_len: int = None, 
                            max_len: int = None, 
                            n_traces: int = None, 
                            random_seed: int = None) -> pd.DataFrame:
    """
    Loads, preprocesses, and filters an event log.

    Parameters:
    - df_name (str): The filename of the CSV file to load.
    - min_len (int, optional): Minimum allowed length of traces for filtering.
    - max_len (int, optional): Maximum allowed length of traces for filtering.
    - n_traces (int, optional): Number of traces to include after filtering.
    - random_seed (int, optional): Seed for reproducible trace selection.

    Returns:
    - pd.DataFrame: The preprocessed and filtered event log as a pandas DataFrame.
    """
    # Load the DataFrame from CSV
    df = pd.read_csv(df_name)
    
    # Keep only the columns 'case:concept:name' and 'case:name', dropping others
    columns_to_keep = ['case:concept:name', 'concept:name']
    df = df[columns_to_keep]
    
    # Map strings to string numbers (assuming this function is defined elsewhere)
    df, map_dict = map_to_string_numbers(df, map_strings_to_integer_strings=True, return_map_dict=True)
    
    # Assign unique numeric identifiers to each case
    df['case:concept:name'] = df.groupby('case:concept:name').ngroup().astype(str)
    
    # Filter the DataFrame based on trace length and number of traces
    df_filtered = filter_log_efficiently(df, min_len=min_len, max_len=max_len, n_traces=n_traces, random_seed=random_seed)
    
    return df_filtered, map_dict


def compare_window_based_baselines(
    df_name: Union[str, pd.DataFrame] = '',
    model_path: str = '',    
    n_train_traces: int = None,
    n_test_traces: int = None,
    train_cases: List[str] = None,
    test_cases: List[str] = None,
    window_lengths_lst: List[int] = None,
    n_final_markings_lst: List[int] = None,
    only_return_model: bool = False,
    window_overlap: int = 0,
    read_model_from_file: bool = False,
    non_sync_penalty: int = 1,
    allow_intersection: bool = True,
    cost_function: Any = None,
    use_heuristics=False,
    max_len=None,
    min_len=None,
    n_traces=None,
    random_seed=304,
    map_dict=None,
    explor_reward=1e-10
) -> Union[pd.DataFrame, Tuple]:

    # Check if df_name is a DataFrame, if not, preprocess it
    if isinstance(df_name, pd.DataFrame):
        df = df_name
    else:
        df, map_dict = load_and_preprocess_log(df_name, min_len=min_len, max_len=max_len, n_traces=n_traces, random_seed=random_seed)

    cost_function = select_cost_function(cost_function)
    np.random.seed(random_seed)
    
    result = train_test_log_split(
        df, 
        n_train_traces=n_train_traces, 
        n_test_traces=n_test_traces,
        train_traces=train_cases,
        test_traces=test_cases,
        random_selection=(train_cases is None and test_cases is None), 
        random_seed=random_seed,
        allow_intersection=allow_intersection
    )
    
    train_df, test_df = result['train_df'], result['test_df']
    test_df['probs'] = [[1.0] for _ in range(len(test_df))]
    
    if read_model_from_file:
        if not model_path or not map_dict:
            raise ValueError("Both 'model_path' and 'map_dict' must be provided when reading a model from a file.")
        
        model = generate_model_from_file(
            model_path,
            activity_mapping_dict=map_dict,
            return_markings=False
        )
    else:
        model = prepare_model(train_df, non_sync_penalty)

    if use_heuristics:
        model = add_transition_mappings_to_model(model)
    
    results = wcb_perform_conformance_checking(
        test_df, model, window_lengths_lst, n_final_markings_lst,
        explor_reward, window_overlap, use_heuristics,
        cost_function
    )

    res_df = wcb_convert_to_dataframe(results)
    return res_df


def wcb_perform_conformance_checking(
    test_df: pd.DataFrame, 
    net: Any,  # Assuming PetriNet is some class defined elsewhere
    window_lengths_lst: List[int],
    n_final_markings_lst: List[int],
    explor_reward: float, 
    window_overlap: int,
    use_heuristics: bool, 
    cost_function: Any
) -> Dict[str, defaultdict]:
    
    results = defaultdict(lambda: defaultdict(list))
    
    trace_dict, lookup_dict = group_similar_traces(test_df)
    total_traces = len(trace_dict)

    for window_len in window_lengths_lst:
        for n_markings in n_final_markings_lst:
            print(f'Evaluating variant: n_markings={n_markings}, window_len={window_len}')
            
            for idx, (trace, cases) in enumerate(trace_dict.items(), 1):
                print(f'\rComputing trace {idx}/{total_traces}', end='')

                representative_case = cases[0]
                trace_df = test_df[test_df['case:concept:name'] == representative_case]
                
                start_time = time.time()
                dist, full_alignment, nodes_opened = horizon_based_conformance(
                    net, trace_df, window_len=window_len, n_unique_final_markings=n_markings,
                    explor_reward=explor_reward, window_overlap=window_overlap,
                    cost_function=cost_function, use_heuristics=use_heuristics
                )
                computation_time = time.time() - start_time
                
                for case in cases:
                    wcb_update_results(case, results, n_markings, window_len, dist, nodes_opened, computation_time, full_alignment)
            
            if window_len is None:
                break

        print()

    return results


def wcb_update_results(case, results, n_markings, window_len, dist, nodes_opened, computation_time, full_alignment):
    key = f'window_{window_len}_markings_{n_markings}'
    results[key]['case_id'].append(case)
    results[key]['cost'].append(dist)
    results[key]['time'].append(computation_time)
    results[key]['nodes_opened'].append(nodes_opened)
    results[key]['alignment'].append(full_alignment)


def wcb_convert_to_dataframe(data):
    """
    Converts a nested dictionary with multiple outer keys into a pandas DataFrame.
    Rounds the 'cost' and 'time' columns to 5 decimal places and prefixes these columns with the outer key.
    Retains only 'case_id', 'cost', 'time', and 'nodes_opened' columns.
    
    Args:
        data (dict): A dictionary where the keys are outer keys and the values are dictionaries containing lists.
    
    Returns:
        pd.DataFrame: A DataFrame with prefixed columns for each outer key.
    """
    all_dfs = []
    for key, inner_dict in data.items():
        # Retain only the relevant columns
        filtered_dict = {
            'case_id': inner_dict.get('case_id'),
            'cost': inner_dict.get('cost'),
            'time': inner_dict.get('time'),
            'nodes_opened': inner_dict.get('nodes_opened')
        }
        
        # Convert the filtered dictionary to a DataFrame
        df = pd.DataFrame(filtered_dict)
        
        # Round the 'cost' and 'time' columns to 5 decimal places, if they exist
        if 'cost' in df.columns:
            df['cost'] = df['cost'].round(5)
        if 'time' in df.columns:
            df['time'] = df['time'].round(5)
        
        # Prefix columns with the outer key, except for 'case_id'
        df = df.rename(columns={
            'cost': f'{key}_cost',
            'time': f'{key}_time',
            'nodes_opened': f'{key}_nodes_opened'
        })
        
        # Append the DataFrame to the list
        all_dfs.append(df)
    
    # Merge all DataFrames on 'case_id'
    final_df = pd.concat(all_dfs, axis=1)
    
    # Remove duplicate 'case_id' columns that might appear due to concatenation
    final_df = final_df.loc[:,~final_df.columns.duplicated()]
    
    return final_df


def run_algorithm_with_timeout(sync_prod, algorithm: str, timeout: Optional[int] = None) -> Optional[Dict[str, Any]]:
    """
    Runs a specified search algorithm on the provided sync product with a timeout.
    Handles exceptions and returns None if the algorithm fails or times out.
    """
    result = {}
    
    def target():
        try:
            if algorithm == 'astar':
                result['alignment'], result['cost'], result['nodes_popped'] = sync_prod.astar_search()
            elif algorithm == 'astar_extended':
                result['alignment'], result['cost'], result['nodes_popped'] = sync_prod.astar_incremental()
            elif algorithm == 'reach':
                result['alignment'], result['cost'], result['nodes_popped'] = sync_prod.reach_search()
        except Exception as e:
            result['error'] = e
    
    thread = Thread(target=target)
    thread.start()
    thread.join(timeout=timeout)
    
    if thread.is_alive():
        print(f" Timeout occurred for {algorithm}.")
        return None
    
    if 'error' in result:
        print(f"Error occurred for {algorithm}: {result['error']}")
        return None
    
    return {'cost': result.get('cost'), 'nodes_popped': result.get('nodes_popped')}


def process_pickle_file(pickle_file_path):
    """
    Process the data from a pickle file to create a DataFrame and print trace information.

    Args:
        pickle_file_path (str): Path to the pickle file.

    Returns:
        pd.DataFrame: DataFrame containing the processed data.
    """
    # Open the file in binary read mode
    with open(pickle_file_path, 'rb') as file:
        # Load the content of the file into a Python object
        data = pickle.load(file)

    # Print the number of unique traces
    print(f'number of unique traces={count_unique_arrays(data)}')

    rows_list = []

    # Iterate through the array of arrays
    for case_index, case_activities in enumerate(data):
        # Iterate through each activity in the case
        for activity in case_activities:
            # Append a tuple (case_id, activity) to the list
            rows_list.append((str(case_index), str(activity)))

    # Create a DataFrame from the list of tuples
    df = pd.DataFrame(rows_list, columns=['case:concept:name', 'concept:name'])

    # Print trace lengths list from DataFrame
    print_trace_lengths_list_from_df(df)

    return df


def save_results(results, filepath):
    """
    Save the results to a CSV file.

    Parameters:
    results (Any): The results data that you want to save. It should be convertible to a pandas DataFrame.
    filepath (str): The file path where the results should be saved, including the filename and .csv extension.
    """
    # Assuming `results` is in a format that can be directly converted to a DataFrame
    df = pd.DataFrame(results)
    
    # Save the DataFrame to a CSV file
    df.to_csv(filepath, index=False)
    print(f"Results successfully saved to {filepath}")