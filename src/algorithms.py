import time
import random
import multiprocessing
import os

from .dataset import Dataset
from .itemset import Itemset
from .utils import generate_valid_candidates, format_time
from .logger import Logger

def standard_apriori(dataset: Dataset, epsilon: float, frontier_only: bool, log: Logger):
    """
    Apriori Algorithm for Quantitative Itemsets 
    """
    # Result set
    R: set[Itemset] = set()
    
    
    # Check if I0 is supported
    dataset.I_0.support = dataset.calculate_support(dataset.I_0)
    if dataset.I_0.support < epsilon:
        # Empty if base is not supported
        return R 
    
    # Supported Witnesses at the previous level
    SW_prev: set[Itemset] = {dataset.I_0}
    # TODO: verify if we need to add I0 to R
    R.add(dataset.I_0) 

    k = 1
    # while SW_{k-1} is not empty
    while SW_prev:
        log.info(f"Level {k}, candidates generation... ")
        start_time = time.time()
        
        # Supported Witnesses at current level
        SW_current: set[Itemset] = set()

        # Set of items to remove from R
        to_remove_from_frontier: set[Itemset] = set()

        # Generate valid candidates from previous level using a Generator
        for cand, parents in generate_valid_candidates(dataset.i0_limits, SW_prev):
            cand.support = dataset.calculate_support(cand)
            
            if cand.support >= epsilon:
                SW_current.add(cand)
                R.add(cand)

            if frontier_only:
                    # If this child is supported, its parents are no longer the limit.
                    to_remove_from_frontier.update(parents)

        if frontier_only and to_remove_from_frontier:
            # remove all non-frontier parents from R
            R.difference_update(to_remove_from_frontier)

        # Update latest level of supported itemsets
        SW_prev = SW_current

        end_time = time.time()
        duration = format_time(end_time - start_time)
        log.info(f"Level {k}: evaluated in {duration}.")
        log.debug(f"Level {k}: {len(SW_current)} supported itemsets found.")
        
        k += 1
    log.info(f"Total itemsets found: {len(R)}")

    return R

def randomic_apriori(dataset: Dataset, epsilon: float, frontier_only: bool, log: Logger, max_iter: int = 0):
    """
    Randomic Apriori: Explores the lattice randomly/deeply instead of level-by-level.
    max_iter: 0 means run until completion. >0 stops after N evaluations.
    """
    R: set[Itemset] = set()
    
    # Check I_0
    dataset.I_0.support = dataset.calculate_support(dataset.I_0)
    if dataset.I_0.support < epsilon:
        return R
    
    R.add(dataset.I_0)
    
    pending = [dataset.I_0]
    # Local Not Supported
    LNS: set[Itemset] = set()
    
    # Visited (To avoid adding the same candidate to pending twice)
    # Since we don't have levels (like in standard apriori), we might generate the same child from different parents
    visited: set[Itemset] = {dataset.I_0}

    count = 0
    
    while pending:
        # Stop condition
        if max_iter > 0 and count >= max_iter:
            log.info(f"Max iterations ({max_iter}) reached.")
            break
            
        # Pick a random index
        idx = random.randrange(len(pending))
        
        # Swap with last and pop
        pending[idx], pending[-1] = pending[-1], pending[idx]
        current: Itemset = pending.pop()
        
        # Generate Candidates
        children = current.get_successors()
        
        for cand in children:
            if cand in visited:
                continue
                
            # Check if this candidate is already known to be bad
            if cand in LNS:
                visited.add(cand)
                continue

            cand.support = dataset.calculate_support(cand)
            visited.add(cand)
            count += 1
            
            if cand.support >= epsilon:
                # Add to Results
                R.add(cand)
                
                # Add to Pending (to go deeper)
                pending.append(cand)
                
                # Frontier Management
                if frontier_only:
                    if current in R:
                        R.remove(current)
            else:
                # Mark as Not Supported
                LNS.add(cand)
        
        if count % 1000 == 0:
            log.debug(f"Evaluated {count} itemsets. Pending: {len(pending)}. R: {len(R)}")

    log.info(f"Random Apriori finished. Evaluated {count} itemsets. Found {len(R)} frequent.")
    return R


def worker_task(dataset: Dataset, epsilon: float, frontier_only: bool, GP, GNS, R, visited, lock):
    """
    Worker process for Distributed Apriori.
    GP: Global Pending (Shared Dict)
    GNS: Global Not Supported (Shared Dict)
    R: Result Set (Shared Dict)
    visited: Global Visited (Shared Dict)
    """
    random.seed(os.getpid())
    
    local_visited = set()

    while True:
        batch = []
        with lock:
            if not GP:
                # OPTIONAL: Wait a bit to see if other workers produce data
                # But for simplicity, we break if empty, acknowledging the race condition
                break
            
            keys = list(GP.keys())
            # Grab up to 50 items
            n_fetch = min(len(keys), 50) 
            for _ in range(n_fetch):
                k = keys.pop() # randomness is less important in batch
                batch.append(k)
                del GP[k]
        
        if not batch:
            break
        
        local_new_pending = []
        local_new_results = []
        local_remove_from_R = []
        
        for item_tuple in batch:
            # Reconstruct Itemset object from tuple
            current_item = Itemset(item_tuple)
            
            # Generate Children
            children = current_item.get_successors()
            
            for cand in children:
                tup = cand.intervals
                
                # Check local cache first
                if tup in local_visited:
                    continue
                local_visited.add(tup)
                
                # Calculate Support
                cand.support = dataset.calculate_support(cand)
                
                if cand.support >= epsilon:
                    local_new_results.append(cand)
                    local_new_pending.append(cand)

                if frontier_only:
                        local_remove_from_R.append(item_tuple)

        # Update Global structures in lock
        with lock:
            # Add new results
            for item in local_new_results:
                R[item.intervals] = item 
            
            # Add to global pending and visited
            for item in local_new_pending:
                if item.intervals not in visited:
                    GP[item.intervals] = True
                    visited[item.intervals] = True
            
            # Remove non-frontier parents
            if frontier_only:
                for parent_tuple in local_remove_from_R:
                    # We use .pop(k, None) to avoid KeyErrors if another worker 
                    # already removed it or if it wasn't in R for some reason.
                    R.pop(parent_tuple, None)

def distributed_apriori(dataset: Dataset, epsilon: float, frontier_only: bool, log: Logger, n_workers: int = 4):
    """
    Randomic Distributed Apriori.
    """
    # Setup Shared Manager
    with multiprocessing.Manager() as manager:
        # Shared Dictionaries (Act as Sets)
        # We use dicts because manager.list() or manager.Value() is harder for sets.
        # Key = Itemset Tuple, Value = True/Itemset
        
        GP = manager.dict()      # Global Pending
        GNS = manager.dict()     # Global Not Supported
        R = manager.dict()       # Relation / Result Set
        visited = manager.dict() # Global Visited
        lock = manager.Lock()
        
        # Initialization
        dataset.I_0.support = dataset.calculate_support(dataset.I_0)
        if dataset.I_0.support < epsilon:
            return set()
        
        # Add I_0 to start
        start_tuple = dataset.I_0.intervals
        GP[start_tuple] = True
        R[start_tuple] = dataset.I_0
        visited[start_tuple] = True
        
        # Start Workers
        processes = []
        for _ in range(n_workers):
            p = multiprocessing.Process(
                target=worker_task,
                args=(dataset, epsilon, frontier_only, GP, GNS, R, visited, lock)
            )
            p.start()
            processes.append(p)
            
        # Wait for completion
        for p in processes:
            p.join()
            
        log.info("Workers finished.")
        
        # Convert Shared Dict back to Set[Itemset]
        final_result = set()
        for item in R.values():
            final_result.add(item)
                
        return final_result