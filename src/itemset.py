from typing_extensions import Self

class Itemset:
    """
    Itemset class with intervals for each attribute.
    """
    __slots__ = ['_intervals', '_support', '_hash_cache']
    
    def __init__(self, intervals: tuple):
        # intervals is a tuple of (start, end) for each attribute index
        self._intervals: tuple = intervals
        self._support = None
        self._hash_cache = None

    def __hash__(self):
        if self._hash_cache is None:
            self._hash_cache = hash(self.intervals)
        return self._hash_cache

    def __eq__(self, other_itemset):
        if not isinstance(other_itemset, Itemset):
            raise TypeError('Cannot compare Itemset with different type')
        
        return self._intervals == other_itemset._intervals

    def __len__(self):
        return len(self._intervals)

    def __str__(self):
        return '-'.join([f'{attr}:{int}' for attr, int in self._intervals.items()])
    
    @property
    def intervals(self) -> tuple:
        return self._intervals
    
    @property
    def support(self) -> float:
        return self._support
    
    @support.setter
    def support(self, value: float):
        self._support = value
    
    def get_successors(self) -> set[Self]:
        """
        Generates itemsets that are one step more specific (shrunk by 1).
        """
        successors = set()
        for i, (col_idx, min_key, max_key) in enumerate(self._intervals):
            # Only shrink if there is space (width >= 2)
            if min_key < max_key - 1:
                base_list = list(self._intervals)
                # Left Shrink
                base_list[i] = (col_idx, min_key + 1, max_key)
                successors.add(Itemset(tuple(base_list)))
                # Right Shrink
                base_list[i] = (col_idx, min_key, max_key - 1)
                successors.add(Itemset(tuple(base_list)))

        return successors
    
    def get_predecessors(self, i0_dict: dict) -> set[Self]:
        """
        Generates itemsets that are one step more general (expanded by 1).
        max_intervals are needed to avoid expanding beyond initial limits.
        """
        predecessors = set()
        for i, (col_idx, min_key, max_key) in enumerate(self._intervals):
            limit_min, limit_max = i0_dict[col_idx]

            # Expand the interval
            if min_key > limit_min:
                base_list = list(self._intervals)
                base_list[i] = (col_idx, min_key - 1, max_key)
                predecessors.add(Itemset(tuple(base_list)))
            
            if max_key < limit_max:
                base_list = list(self._intervals)
                base_list[i] = (col_idx, min_key, max_key + 1)
                predecessors.add(Itemset(tuple(base_list)))
                
        return predecessors
    
    def describe(self, dataset) -> str:
        """
        Returns a human-readable string using real values.
        Accepts the 'dataset' object to access id_to_col_name and attr_map.
        """
        parts = []
        for col_id, min_key, max_key in self.intervals:
            
            # Get Name
            col_name = dataset.id_to_col_name[col_id]
            
            # Get Values
            val_min = dataset.attr_map[col_name].get(min_key, float('-inf'))
            val_max = dataset.attr_map[col_name].get(max_key, float('inf'))
            
            parts.append(f"{col_name}:[{val_min:.2f}, {val_max:.2f}]")
                
        return ", ".join(parts)