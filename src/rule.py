from __future__ import annotations

from .itemset import Itemset
from .logger import Logger

from typing import List, Set, TYPE_CHECKING
from scipy.stats import fisher_exact
from functools import lru_cache

if TYPE_CHECKING:
    from .dataset import Dataset

import multiprocessing
import itertools

# Global state for worker processes
_worker_dataset: Dataset = None
_worker_min_conf: float = 0.0
_worker_support_cache = None

def _init_worker(dataset: Dataset, min_conf: float):
    """
    Initializes the worker process with dataset and min_conf.
    """
    global _worker_dataset, _worker_min_conf, _worker_support_cache
    _worker_dataset = dataset
    _worker_min_conf = min_conf
    
    # Ensure every core has its own fast cache
    @lru_cache(maxsize=50000)
    def cached_support(intervals_tuple: tuple) -> float:
        temp = Itemset(intervals_tuple)
        return _worker_dataset.calculate_support(temp)
    
    _worker_support_cache = cached_support

def _calculate_p_value(supp_union: float, supp_ant: float, supp_cons: float, N: int) -> float:
    """
    Calculates the p-value using Fisher's Exact Test.
    Contingency Table:
                  | Cons     | Not Cons
        ----------|----------|----------
        Ant       | a (X&Y)  | b (X&!Y)
        Not Ant   | c (!X&Y) | d (!X&!Y)
    """
    n_union = int(supp_union * N)
    n_ant = int(supp_ant * N)
    n_cons = int(supp_cons * N)
    
    a = n_union
    b = n_ant - n_union
    c = n_cons - n_union
    d = N - n_ant - c
    
    _, p_value = fisher_exact([[a, b], [c, d]], alternative='greater')
    return p_value

def _extract_parallel_batch(itemsets_batch: List[Itemset]) -> List[AssociationRule]:
    """
    Processes a batch of itemsets to extract association rules.
    """
    rules = []
    
    # Access global state
    dataset = _worker_dataset
    min_conf = _worker_min_conf
    get_support = _worker_support_cache
    
    for union_itemset in itemsets_batch:
        all_intervals = union_itemset.intervals
        n_items = len(all_intervals)
        
        if n_items < 2:
            continue

        # Get Union Support
        support_union = union_itemset.support
        if support_union is None:
            support_union = get_support(all_intervals)

        # Iterate all Antecedent combinations
        for i in range(1, n_items):
            for ant_indices in itertools.combinations(range(n_items), i):
                
                # Antecedent 
                antecedent_data = tuple(all_intervals[k] for k in ant_indices)
                support_antecedent = get_support(antecedent_data)
                if support_antecedent == 0: continue

                # Check Confidence
                confidence = support_union / support_antecedent

                if confidence >= min_conf:
                    antecedent = Itemset(tuple(all_intervals[k] for k in ant_indices))
                    consequent = Itemset(tuple(all_intervals[k] for k in range(n_items) if k not in ant_indices))

                    antecedent.support = support_antecedent
                    consequent.support = get_support(consequent.intervals)

                    # Metrics
                    lift = confidence / consequent.support if consequent.support > 0 else 0.0
                    p_val = _calculate_p_value(support_union, antecedent.support, consequent.support, len(dataset.data))

                    rule = AssociationRule(
                        antecedent=antecedent,
                        consequent=consequent,
                        support_union=support_union,
                        support_antecedent=antecedent.support,
                        support_consequent=consequent.support,
                        confidence=confidence,
                        lift=lift,
                        p_value=p_val
                    )

                    rules.append(rule)
    return rules

class AssociationRule:
    """
    Represents a Quantitative Association Rule: Antecedent -> Consequent
    """
    def __init__(self, antecedent: Itemset, consequent: Itemset, 
                 support_union: float, support_antecedent: float, support_consequent: float,
                 confidence: float, lift: float, p_value: float):
        self.antecedent: Itemset = antecedent
        self.consequent: Itemset = consequent
        self.support_union: float = support_union
        self.support_antecedent: float = support_antecedent
        self.support_consequent: float = support_consequent
        self.confidence: float = confidence
        self.lift: float = lift
        self.p_value: float = p_value
    
    def describe(self, dataset: Dataset) -> str:
        """
        Returns the rule with translated values.
        """
        ant_str = self.antecedent.describe(dataset)
        cons_str = self.consequent.describe(dataset)
        
        return (
            f"{ant_str} => {cons_str}\n"
            f"   [Conf: {self.confidence:.2f} | Lift: {self.lift:.2f} | P-Val: {self.p_value:.4e}]"
        )
    
    def __str__(self):
        return f"{self.antecedent} => {self.consequent} (Lift: {self.lift:.2f})"
    
class RuleExtractor:
    def __init__(self, dataset: Dataset, min_conf: float = 0.8, log: Logger = None):
        self.dataset = dataset
        self.min_conf = min_conf
        self.log = log

    def extract_rules(self, frontier: Set[Itemset], n_workers: int = 4) -> List[AssociationRule]:
        """
        Extracts rules in parallel.
        """
        frontier_list = list(frontier)
        total_items = len(frontier_list)
        
        self.log.info(f"Extracting rules from {total_items} itemsets using {n_workers} workers...")

        # Chunking
        chunk_size = max(1, total_items // (n_workers * 4))
        chunks = [frontier_list[i:i + chunk_size] for i in range(0, total_items, chunk_size)]
        
        # Parallel Execution
        final_rules = []
        
        with multiprocessing.Pool(processes=n_workers, 
                                  initializer=_init_worker, 
                                  initargs=(self.dataset, self.min_conf)) as pool:
            
            # Map the chunks to the workers
            results = pool.map(_extract_parallel_batch, chunks)
            
            # Flatten results
            for batch_rules in results:
                final_rules.extend(batch_rules)
        
        self.log.info(f"Extraction complete. Found {len(final_rules)} valid rules.")
            
        return final_rules
