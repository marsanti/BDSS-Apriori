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
        self.max_p_value: float = max_p_value
        self.min_lift: float = min_lift
        
        self.final_rules: List[AssociationRule] = []
        
        # Ant: Set of (ColID, MinKey, MaxKey)
        self.Ant: Set[Tuple[int, int, int]] = set()
        self.Cons: Set[Tuple[int, int, int]] = set()
        
        self.interval_support_cache = {}
        
    def execute(self, n_permutations: int = 50):
        """
        Main execution flow for Exercise 3.
        """
        # Filter Rules
        self._filter_final_rules()
        
        if not self.final_rules:
            if self.log: self.log.warning("No Final Rules found matching criteria.")
            return

        # Build Ant and Cons sets
        self._build_sets()

        # Pre-calculate supports for all atomic intervals in Ant (Optimization for Cl)
        self._precompute_ant_supports()

        # Compute Shapley Values for each Consequent
        self._compute_shapley_values(n_permutations)

    def _filter_final_rules(self):
        """
        Keep only rules with p-value < self.max_p_value and lift > self.min_lift.
        """
        self.final_rules = [
            r for r in self.all_rules 
            if r.p_value < self.max_p_value and r.lift > self.min_lift
        ]
        if self.log:
            self.log.info(f"Filtered Final Rules: {len(self.final_rules)} (from {len(self.all_rules)})")

    def _build_sets(self):
        """
        Constructs Ant and Cons sets from final rules.
        """
        for rule in self.final_rules:
            for col_id, min_k, max_k in rule.antecedent.intervals:
                self.Ant.add((col_id, min_k, max_k))
            
            for col_id, min_k, max_k in rule.consequent.intervals:
                self.Cons.add((col_id, min_k, max_k))
                
        if self.log:
            self.log.info(f"Unique Intervals - Ant: {len(self.Ant)}, Cons: {len(self.Cons)}")

    def _precompute_ant_supports(self):
        """
        Calculates support for every single interval in Ant.
        """
        if self.log: self.log.info("Pre-computing interval supports...")
        
        batch_candidates = []
        mapping = [] 
        
        for col_id, min_k, max_k in self.Ant:
            # Create a tuple of tuples with 1 element
            item = Itemset(((col_id, min_k, max_k),))
            
            batch_candidates.append(item)
            mapping.append((col_id, min_k, max_k))
            
        if batch_candidates:
            supports = [self.dataset.calculate_support(i) for i in batch_candidates]

            for i, supp in enumerate(supports):
                self.interval_support_cache[mapping[i]] = supp

    def _cl(self, subset_ant: List[Tuple[int, int, int]]) -> Itemset:
        """
        Conflict Resolution function Cl(Ant').
        Resolves multiple intervals on the same attribute by picking the one with Max Support.
        """
        # Group by Column ID
        grouped = defaultdict(list)
        for interval in subset_ant:
            col_id = interval[0]
            grouped[col_id].append(interval)
            
        final_list = []
        
        for col_id, candidates in grouped.items():
            # Find candidate with max support
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
            
            final_list.append(best_cand)
            
        final_list.sort(key=lambda x: x[0])
        
        return Itemset(tuple(final_list))

    def _j_measure(self, antecedent: Itemset, consequent: Itemset) -> float:
        """
        Calculates J-Measure for rule Ant -> Cons.
        J = p(Ant) * CrossEntropy(Cons|Ant || Cons)
        """
        # Calculate Supports
        antecedent.support = self.dataset.calculate_support(antecedent)
        if antecedent.support == 0: return 0.0
        
        consequent.support = self.dataset.calculate_support(consequent)
        
        # Create Union by merging tuples and sorting
        union_list = list(antecedent.intervals) + list(consequent.intervals)
        union_list.sort(key=lambda x: x[0])
        union_itemset = Itemset(tuple(union_list))
        
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
        Computes Approximate Shapley Values for each (j, interval) in Cons.
        """
        if self.log: self.log.info(f"Computing Shapley Values (M={M} permutations)...")

        for j_idx, cons_tuple in enumerate(self.Cons):
            # cons_tuple is (col_id, min, max)
            cons_col_id = cons_tuple[0]
            
            cons_itemset = Itemset((cons_tuple,))
            
            # Filter Ant excluding this column ID
            Ant_j = [x for x in self.Ant if x[0] != cons_col_id]
            
            if not Ant_j:
                continue

            # Accumulator for Shapley values: {interval_tuple: sum_marginal_contribution}
            shapley_sums = defaultdict(float)
            
            for m in range(M):
                perm = list(Ant_j)
                random.shuffle(perm)
                
                S_current = []
                v_current = 0.0 # v(Empty Set) is usually 0 for J-Measure implies no info
                
                for interval in perm:
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
        
        # Helper to format string
        def fmt(interval):
            col_id, mn, mx = interval
            name = self.dataset.id_to_col_name[col_id]
            v_min = self.dataset.attr_map[name].get(mn, "?")
            v_max = self.dataset.attr_map[name].get(mx, "?")
            return f"{name}:[{v_min:.2f}, {v_max:.2f}]"

        if self.log:
            c_desc = fmt(cons_tuple)
            self.log.info(f"\n--- Shapley Analysis for Consequent: {c_desc} ---")
            self.log.info("Top 5 Contributors:")
            
            for i, (interval, score) in enumerate(sorted_shapley[:5]):
                self.log.info(f"   {i+1}. {fmt(interval)} | Shapley: {score:.4f}")