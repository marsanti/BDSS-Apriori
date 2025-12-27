from __future__ import annotations

from numba import njit
from .itemset import Itemset
from .rule import AssociationRule
from .logger import Logger
from typing import Generator, Tuple, Set, List, TYPE_CHECKING

if TYPE_CHECKING:
    from .dataset import Dataset

# This runs at C-speed. It checks all conditions in a single pass per row.
@njit
def compute_support_numba(data_matrix, col_indices, bounds):
    n_rows = data_matrix.shape[0]
    n_constraints = len(col_indices)
    count = 0
    
    for i in range(n_rows):
        match = True
        for j in range(n_constraints):
            col_idx = col_indices[j]
            val = data_matrix[i, col_idx]
            
            # bounds[j, 0] is min_val, bounds[j, 1] is max_val
            # Condition: min_val < val < max_val
            if not (bounds[j, 0] < val < bounds[j, 1]):
                match = False
                break
        
        if match:
            count += 1
            
    return count / n_rows

def generate_valid_candidates(I_0: Itemset, SW_prev: Set[Itemset]) -> Generator[Tuple[Itemset, Set[Itemset]], None, None]:
    """
    Generator of valid candidates.
    Yields (candidate, parents) lazily to save RAM.
    """
    i0_intervals = I_0.intervals

    for item in SW_prev:
        for cand in item.get_successors():

            # Get Parents
            parents = cand.get_predecessors(i0_intervals)

            # Pruning
            # checks if ALL parents are currently supported
            is_witness = all(p in SW_prev for p in parents)
            
            if is_witness:
                yield cand, parents

def show_rules(rules: List[AssociationRule], dataset: Dataset, log: Logger, n_rules: int = 10):
    """
    Displays n rules in a readable format.
    """
    log.info(f"Top {n_rules} High-Lift Rules:")
    # safety check
    n_rules = min(n_rules, len(rules))
    
    for i, rule in enumerate(rules[:n_rules]):
        log.info(f"Rule #{i+1}:\n{rule.describe(dataset.attr_map)}")

def format_time(seconds: float) -> str:
    """
    Formats time in seconds to a human-readable string.
    """
    hours = int(seconds // 3600)
    seconds -= hours * 3600
    minutes = int(seconds // 60)
    seconds -= minutes * 60
    milliseconds = int((seconds - int(seconds)) * 1000)
    
    formatted = ""

    if hours > 0:
        formatted += f"{hours}h "
    if minutes > 0 or hours > 0:
        formatted += f"{minutes}m "
    if int(seconds) > 0 or minutes > 0 or hours > 0:
        formatted += f"{int(seconds)}s "
    if milliseconds > 0 and minutes < 0 and hours < 0:
        # Only show milliseconds if less than a minute
        formatted += f"{milliseconds}ms"

    return formatted
    