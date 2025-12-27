from __future__ import annotations
import math
import random
from typing import List, Set, Tuple, TYPE_CHECKING
from collections import defaultdict

# Avoid circular imports at runtime
if TYPE_CHECKING:
    from .dataset import Dataset
    from .rule import AssociationRule
    from .logger import Logger

from .itemset import Itemset

class ShapleyAnalyzer:
    def __init__(self, dataset: Dataset, rules: List[AssociationRule], log: Logger = None, max_p_value: float = 0.05, min_lift: float = 1.5):
        self.dataset = dataset
        self.all_rules = rules
        self.log = log
        self.max_p_value: float = 0.05
        self.min_lift: float = 1.5
        
        self.final_rules: List[AssociationRule] = []
        
        # Ant: Set of (AttributeName, MinKey, MaxKey)
        # We store keys (indices) instead of raw values to match Itemset structure
        self.Ant: Set[Tuple[str, int, int]] = set()
        self.Cons: Set[Tuple[str, int, int]] = set()
        
        # Cache single interval supports for Cl() function speedup
        # {(attr, min, max): support}
        self.interval_support_cache = {}
        
    def execute(self, n_permutations: int = 50):
        """
        Main execution flow for Exercise 3.
        """
        # Filter Rules
        self._filter_final_rules()
        
        if not self.final_rules:
            if self.log: self.log.warning("No Final Rules found matching criteria (p<0.05, lift>1.5).")
            return

        # Build Ant and Cons sets
        self._build_sets()
        
        # Pre-calculate supports for all atomic intervals in Ant (Optimization for Cl)
        self._precompute_ant_supports()
        
        # Compute Shapley Values for each Consequent
        self._compute_shapley_values(n_permutations)

    def _filter_final_rules(self):
        """
        Keep only rules with p-value < 0.05 and lift > 1.5[cite: 145, 146].
        """
        self.final_rules = [
            r for r in self.all_rules 
            if r.p_value < self.max_p_value and r.lift > self.min_lift
        ]
        if self.log:
            self.log.info(f"Filtered Final Rules: {len(self.final_rules)} (from {len(self.all_rules)})")

    def _build_sets(self):
        """
        Constructs Ant and Cons sets from final rules[cite: 149, 150].
        """
        for rule in self.final_rules:
            # Add Antecedent intervals
            for attr, (min_k, max_k) in rule.antecedent.intervals.items():
                self.Ant.add((attr, min_k, max_k))
            
            # Add Consequent intervals
            for attr, (min_k, max_k) in rule.consequent.intervals.items():
                self.Cons.add((attr, min_k, max_k))
                
        if self.log:
            self.log.info(f"Unique Intervals - Ant: {len(self.Ant)}, Cons: {len(self.Cons)}")

    def _precompute_ant_supports(self):
        """
        Calculates support for every single interval in Ant.
        Needed for the Cl() maximization step[cite: 155].
        """
        if self.log: self.log.info("Pre-computing interval supports...")
        
        # Create 1-item Itemsets and calc support
        batch_candidates = []
        mapping = [] # to map back results
        
        for attr, min_k, max_k in self.Ant:
            item = Itemset({attr: (min_k, max_k)})
            batch_candidates.append(item)
            mapping.append((attr, min_k, max_k))
            
        # Use batch calculation for speed
        if batch_candidates:
            supports = self.dataset.calculate_support_batch(batch_candidates)
            for i, supp in enumerate(supports):
                self.interval_support_cache[mapping[i]] = supp

    def _cl(self, subset_ant: List[Tuple[str, int, int]]) -> Itemset:
        """
        Conflict Resolution function Cl(Ant')[cite: 154].
        Resolves multiple intervals on the same attribute by picking the one with Max Support.
        """
        # Group by attribute
        grouped = defaultdict(list)
        for interval in subset_ant:
            attr = interval[0]
            grouped[attr].append(interval)
            
        final_intervals = {}
        
        for attr, candidates in grouped.items():
            # Find candidate with max support [cite: 155]
            best_cand = None
            max_supp = -1.0
            
            for cand in candidates:
                # Use cached support
                s = self.interval_support_cache.get(cand, 0.0)
                
                # Tie-breaking: Lexicographical on (min, max) implied by tuple comparison
                if s > max_supp:
                    max_supp = s
                    best_cand = cand
                elif s == max_supp:
                    # Explicit tie-break on bounds
                    if best_cand is None or cand > best_cand:
                        best_cand = cand
            
            # Add winner to itemset definition
            # best_cand is (attr, min, max)
            final_intervals[attr] = (best_cand[1], best_cand[2])
            
        return Itemset(final_intervals)

    def _j_measure(self, antecedent: Itemset, consequent: Itemset) -> float:
        """
        Calculates J-Measure for rule Ant -> Cons[cite: 163].
        J = p(Ant) * CrossEntropy(Cons|Ant || Cons)
        """
        # Calculate Supports
        antecedent.support = self.dataset.calculate_support(antecedent)
        if antecedent.support == 0: return 0.0
        
        consequent.support = self.dataset.calculate_support(consequent)
        
        # Union support
        # We need to construct the union itemset to calculate support(Ant U Cons)
        union_intervals = antecedent.intervals.copy()
        union_intervals.update(consequent.intervals) # If overlap, Cons overwrites (should not happen in valid rule)
        union_itemset = Itemset(union_intervals)
        
        union_itemset.support = self.dataset.calculate_support(union_itemset)
        
        # Probabilities
        prob_c_given_a = union_itemset.support / antecedent.support
        prob_c = consequent.support
        
        # J-Measure Formula
        # Term 1: p(c|a) * log(p(c|a) / p(c))
        term1 = 0.0
        if prob_c_given_a > 0 and prob_c > 0:
            term1 = prob_c_given_a * math.log2(prob_c_given_a / prob_c)
            
        # Term 2: (1 - p(c|a)) * log((1 - p(c|a)) / (1 - p(c)))
        term2 = 0.0
        if (1 - prob_c_given_a) > 0 and (1 - prob_c) > 0:
            term2 = (1 - prob_c_given_a) * math.log2((1 - prob_c_given_a) / (1 - prob_c))
            
        return antecedent.support * (term1 + term2)

    def _compute_shapley_values(self, M: int):
        """
        Computes Approximate Shapley Values for each (j, interval) in Cons[cite: 164].
        Uses Monte Carlo permutation sampling.
        """
        if self.log: self.log.info(f"Computing Shapley Values (M={M} permutations)...")

        # Iterate over each target consequent in Cons
        for j_idx, cons_tuple in enumerate(self.Cons):
            cons_attr, cons_min, cons_max = cons_tuple
            
            # Construct Consequent Itemset object
            cons_itemset = Itemset({cons_attr: (cons_min, cons_max)})
            
            # Ant_j: Subset of Ant excluding attribute j [cite: 162]
            # Filters out any interval that belongs to the same attribute as the consequent
            Ant_j = [x for x in self.Ant if x[0] != cons_attr]
            
            if not Ant_j:
                continue

            # Accumulator for Shapley values: {interval_tuple: sum_marginal_contribution}
            shapley_sums = defaultdict(float)
            
            # Monte Carlo Loop
            for m in range(M):
                # Generate Random Permutation of Ant_j
                perm = list(Ant_j)
                random.shuffle(perm)
                
                # Iterate through permutation
                # S is the set of predecessors
                S_current = []
                v_current = 0.0 # v(Empty Set) is usually 0 for J-Measure implies no info
                
                for interval in perm:
                    # Form S U {i}
                    S_next = S_current + [interval]
                    
                    # Compute v(S U {i}) -> J-Measure(Cl(S_next) -> Cons)
                    itemset_next = self._cl(S_next)
                    v_next = self._j_measure(itemset_next, cons_itemset)
                    
                    # Marginal Contribution
                    contribution = v_next - v_current
                    shapley_sums[interval] += contribution
                    
                    # Update for next step
                    v_current = v_next
                    S_current = S_next
            
            # Average and Log Top Contributors
            self._log_shapley_results(cons_tuple, shapley_sums, M)

    def _log_shapley_results(self, cons_tuple, shapley_sums, M):
        """
        Formats and logs the top Shapley values for a specific consequent.
        """
        # Calculate averages
        avg_shapley = {k: v / M for k, v in shapley_sums.items()}
        
        # Sort by value descending
        sorted_shapley = sorted(avg_shapley.items(), key=lambda x: x[1], reverse=True)
        
        # Format Consequent String
        c_attr, c_min, c_max = cons_tuple
        # Try to look up real values if map exists
        c_desc = f"{c_attr}:[{c_min}-{c_max}]"
        if self.dataset.attr_map:
             v_min = self.dataset.attr_map[c_attr].get(c_min, "?")
             v_max = self.dataset.attr_map[c_attr].get(c_max, "?")
             c_desc = f"{c_attr}:[{v_min:.1f}, {v_max:.1f}]"

        if self.log:
            self.log.info(f"\n--- Shapley Analysis for Consequent: {c_desc} ---")
            self.log.info("Top 5 Contributors (Intervals that add most information):")
            
            for i, (interval, score) in enumerate(sorted_shapley[:5]):
                i_attr, i_min, i_max = interval
                i_desc = f"{i_attr}:[{i_min}-{i_max}]"
                if self.dataset.attr_map:
                    v_min = self.dataset.attr_map[i_attr].get(i_min, "?")
                    v_max = self.dataset.attr_map[i_attr].get(i_max, "?")
                    i_desc = f"{i_attr}:[{v_min:.1f}, {v_max:.1f}]"
                
                self.log.info(f"   {i+1}. {i_desc} | Shapley: {score:.4f}")