import pandas as pd
import numpy as np
from typing import Set, Dict

from .itemset import Itemset
from .utils import compute_support_numba

class Dataset:
    """
    Represents a dataset with data and associated itemsets.
    """
    def __init__(self, data: pd.DataFrame, max_bins: int):
        self.data: pd.DataFrame = data
        self.max_bins: int = max_bins

        # optimization step: instead of dict with column names as keys use tuples with first element as column index
        self.col_name_to_id = {name: i for i, name in enumerate(self.data.columns)}
        self.id_to_col_name = {i: name for i, name in enumerate(self.data.columns)}

        self.I_0: Itemset = None
        self.attr_map: Dict[str, Dict[int, float]] = None
        # algorithms result
        self.R: Set[Itemset] = None
        
        self._map_intervals()

    def _map_intervals(self):
        """
        Get intervals to feature values.
        """
        attr_map = {}
        I_0 = [] 

        self.i0_limits = {}

        # Create a list of intervals for each feature
        for attr in self.data.columns:
            if attr != 'Counter':
                # Sort unique values
                # unique_values = sorted(self.data[attr].unique())      
                try:
                    # Create up to self.max_bins bins using qcut
                    _, bins = pd.qcut(self.data[attr], q=self.max_bins, retbins=True, duplicates='drop')
                    unique_values = sorted(bins)
                except Exception:
                    # Try creating linear bins if qcut fails
                    unique_values = sorted(self.data[attr].unique())
                    if len(unique_values) > self.max_bins:
                         unique_values = np.linspace(min(unique_values), max(unique_values), self.max_bins)          
                # Minimum value of the interval
                min_key = 0
                # Map index 0 to value "min - 1"
                attr_map[attr] = {0: min(unique_values) - 1} 


                # Middle values for intervals
                for idx in range(len(unique_values) - 1):
                    # difference between consecutive unique values
                    diff = unique_values[idx + 1] - unique_values[idx]
                    # Map index k to midpoint
                    attr_map[attr].update({(min_key + idx + 1): unique_values[idx] + diff/2})

                # Max value of the intervals
                # Map last index to "max + 1"
                max_key = len(unique_values)
                attr_map[attr].update({len(unique_values): max(unique_values) + 1})

                # example: unique_values for attr = [10, 20, 30]
                # middle points are 15, 25, because 20-10=10 10/2=5 10+5=15 and 30-20=10 10/2=5 20+5=25
                # attr_map[attr] = {0: 9, 1: 15, 2: 25, 3: 31}

                # Init intervals for the feature: (0, N)
                col_id = self.col_name_to_id[attr]
                I_0.append((col_id, min_key, max_key))
                self.i0_limits[col_id] = (min_key, max_key)

        # Attribute idx to value map
        self.attr_map = attr_map
        I_0.sort(key=lambda x: x[0])
        self.I_0 = Itemset(tuple(I_0))

    def calculate_support(self, itemset: 'Itemset') -> float:
        """
        Calculates e-support.
        Sum(C for satisfied tuples) / Total C
        """
        if not hasattr(self, '_data_numpy'):
            self._data_numpy = self.data.values.astype(np.float64)

        intervals = itemset.intervals 
        
        if not intervals:
            return 1.0

        # We still need to map the "Key Index" (0, 1...) to "Real Float Value" (0.5, 12.3...)
        # We can build the bounds array directly
        col_indices = []
        bounds = []
        
        for col_idx, min_key, max_key in intervals:
            col_name = self.id_to_col_name[col_idx]
            
            # Get real values from attr_map
            real_min = self.attr_map[col_name][min_key]
            real_max = self.attr_map[col_name][max_key]
            
            col_indices.append(col_idx)
            bounds.append((real_min, real_max))

        # Pass to Numba
        return compute_support_numba(
            self._data_numpy, 
            np.array(col_indices, dtype=np.int32), 
            np.array(bounds, dtype=np.float64)
        )
    