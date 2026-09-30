from collections import OrderedDict


class LRUCache:
    def __init__(self, capacity):
        if not isinstance(capacity, int) or isinstance(capacity, bool) or capacity <= 0:
            raise ValueError("capacity must be a positive integer")
        self._capacity = capacity
        self._data = OrderedDict()

    @property
    def capacity(self):
        return self._capacity

    def get(self, key, default=None):
        if key not in self._data:
            return default
        self._data.move_to_end(key)
        return self._data[key]

    def put(self, key, value):
        if key in self._data:
            self._data.move_to_end(key)
        self._data[key] = value
        if len(self._data) > self._capacity:
            self._data.popitem(last=False)

    def __len__(self):
        return len(self._data)
