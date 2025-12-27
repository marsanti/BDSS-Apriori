from .dataset import Dataset
from .itemset import Itemset
from .utils import generate_valid_candidates, format_time
from .logger import Logger
import time

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
        for cand, parents in generate_valid_candidates(dataset.I_0, SW_prev):
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
