from typing_extensions import Self

class Itemset:
    """
    Itemset class with intervals for each attribute.
    """
    __slots__ = ['_intervals', '_support', '_hash_cache']
    
    def __init__(self, intervals: dict):
        # intervals is a tuple of (start, end) for each attribute index
        self._intervals: dict = intervals
        self._support = None
        self._hash_cache = None
    
    def __hash__(self):
        if self._hash_cache is None:
            self._hash_cache = hash(frozenset(self._intervals.items()))
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
    def intervals(self) -> dict:
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
        # Safety check
        assert len(self._intervals.keys()) > 0

        successors = set()
        for attr, (min_key, max_key) in self._intervals.items():
            # Only shrink if there is space (width >= 2)
            if min_key < max_key - 1:
                # Left Shrink
                sl_copy = self._intervals.copy()
                sl_copy[attr] = (min_key + 1, max_key)
                successors.add(Itemset(sl_copy))
                # Right Shrink
                sr_copy = self._intervals.copy()
                sr_copy[attr] = (min_key, max_key - 1)
                successors.add(Itemset(sr_copy))

        return successors
    
    def get_predecessors(self, max_intervals: dict) -> set[Self]:
        """
        Generates itemsets that are one step more general (expanded by 1).
        max_intervals are needed to avoid expanding beyond initial limits.
        """
        predecessors = set()
        for attr, (min_key, max_key) in self._intervals.items():
            i0_min_key, i0_max_key = max_intervals[attr]

            # Expand the interval
            if min_key > i0_min_key:
                copy = self._intervals.copy()
                copy[attr] = (min_key - 1, max_key)
                predecessors.add(Itemset(copy))
            
            if max_key < i0_max_key:
                copy = self._intervals.copy()
                copy[attr] = (min_key, max_key + 1)
                predecessors.add(Itemset(copy))
                
        return predecessors
    
    def describe(self, attr_map: dict) -> str:
        """
        Returns a human-readable string using real values from the dataset map.
        Format: Attr:[min_val - max_val]
        """
        parts = []
        # Sort keys for consistent output order
        for attr in sorted(self._intervals.keys()):
            min_key, max_key = self._intervals[attr]
            
            if attr in attr_map:
                # Retrieve real values
                # attr_map structure: {attr_name: {index: value}}
                val_min = attr_map[attr].get(min_key, float('-inf'))
                val_max = attr_map[attr].get(max_key, float('inf'))
                
                parts.append(f"{attr}:[{val_min:.2f}, {val_max:.2f}]")
            else:
                # Fallback if attribute not found in map
                parts.append(f"{attr}:idx[{min_key}-{max_key}]")
                
        return ", ".join(parts)