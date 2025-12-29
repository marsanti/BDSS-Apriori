from __future__ import annotations

from .itemset import Itemset
from .logger import Logger

from typing import List, Set, TYPE_CHECKING
from scipy.stats import fisher_exact

if TYPE_CHECKING:
    from .dataset import Dataset

import itertools

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

    def extract_rules(self, frontier: Set[Itemset]) -> List[AssociationRule]:
        """
        Extracts rules from the provided frontier itemsets.
        """
        if self.log:
            self.log.info(f"Extracting rules from {len(frontier)} frontier itemsets (Min Conf: {self.min_conf})...")

        rules = []
        
        for idx, union_itemset in enumerate(frontier):
            all_intervals = union_itemset.intervals
            n_items = len(all_intervals)            
            # We need at least 2 attributes to make a rule (A -> B)
            if n_items < 2:
                continue

            # Support of the union
            supp_union = union_itemset.support
            if supp_union is None:
                supp_union = self.dataset.calculate_support(union_itemset)

            # Generate all non-empty subsets of indices for Antecedent
            for i in range(1, n_items):
                for ant_indices in itertools.combinations(range(n_items), i):
                    
                    # Create Antecedent and Consequent Itemsets
                    ant_data = [all_intervals[k] for k in ant_indices]
                    cons_data = [all_intervals[k] for k in range(n_items) if k not in ant_indices]
                    
                    antecedent = Itemset(tuple(ant_data))
                    consequent = Itemset(tuple(cons_data))

                    # Calculate Antecedent Support
                    antecedent.support = self.dataset.calculate_support(antecedent)
                    
                    if antecedent.support == 0:
                        continue

                    # Check Confidence
                    confidence = supp_union / antecedent.support
                    
                    if confidence >= self.min_conf:
                        # Calculate Consequent Support
                        consequent.support = self.dataset.calculate_support(consequent)

                        # Calculate Metrics
                        lift = confidence / consequent.support if consequent.support > 0 else 0.0
                        p_val = self._calculate_p_value(supp_union, antecedent.support, consequent.support)

                        rule = AssociationRule(
                            antecedent=antecedent,
                            consequent=consequent,
                            support_union=supp_union,
                            support_antecedent=antecedent.support,
                            support_consequent=consequent.support,
                            confidence=confidence,
                            lift=lift,
                            p_value=p_val
                        )
                        rules.append(rule)
        
        if self.log:
            self.log.info(f"Extraction complete. Found {len(rules)} valid rules.")
            
        return rules
    def _calculate_p_value(self, supp_union: float, supp_ant: float, supp_cons: float) -> float:
        """
        Calculates P-value using Fisher's Exact Test on the contingency table.
        """
        N = len(self.dataset.data)
        
        n_union = int(supp_union * N)       # X and Y
        n_ant = int(supp_ant * N)           # X
        n_cons = int(supp_cons * N)         # Y
        
        # Contingency Table:
        #           | Cons     | Not Cons
        # ----------|----------|----------
        # Ant       | a (X&Y)  | b (X&!Y)
        # Not Ant   | c (!X&Y) | d (!X&!Y)
        
        a = n_union
        b = n_ant - n_union
        c = n_cons - n_union
        d = N - n_ant - c
        
        _, p_value = fisher_exact([[a, b], [c, d]], alternative='greater')
        
        return p_value