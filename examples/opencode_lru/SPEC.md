# Small Python coding sample

Implement an LRUCache in lru_cache.py using only the Python standard library.
- Constructor capacity must be a positive integer; reject invalid capacity with ValueError.
- get(key, default=None) returns the value and marks it most recently used.
- put(key, value) inserts or updates, evicting the least recently used entry when full.
- __len__ returns the number of entries.
Write test_lru_cache.py using unittest to cover insertion, reads, eviction, update order, missing keys, and invalid capacities.
Run python3 -m unittest -v and report its result.
