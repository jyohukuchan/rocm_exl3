import unittest

from lru_cache import LRUCache


class TestLRUCache(unittest.TestCase):
    def test_insertion_and_len(self):
        cache = LRUCache(3)
        self.assertEqual(len(cache), 0)
        cache.put('a', 1)
        cache.put('b', 2)
        self.assertEqual(len(cache), 2)
        cache.put('c', 3)
        self.assertEqual(len(cache), 3)

    def test_get_returns_value(self):
        cache = LRUCache(2)
        cache.put('a', 1)
        self.assertEqual(cache.get('a'), 1)

    def test_get_missing_key_returns_default(self):
        cache = LRUCache(2)
        self.assertIsNone(cache.get('missing'))
        self.assertEqual(cache.get('missing', 'fallback'), 'fallback')
        cache.put('a', 1)
        self.assertEqual(cache.get('b', 0), 0)
        self.assertEqual(len(cache), 1)

    def test_get_marks_most_recently_used(self):
        cache = LRUCache(2)
        cache.put('a', 1)
        cache.put('b', 2)
        cache.get('a')
        cache.put('c', 3)
        self.assertIsNone(cache.get('b'))
        self.assertEqual(cache.get('a'), 1)
        self.assertEqual(cache.get('c'), 3)
        self.assertEqual(len(cache), 2)

    def test_eviction_removes_least_recently_used(self):
        cache = LRUCache(2)
        cache.put('a', 1)
        cache.put('b', 2)
        cache.put('c', 3)
        self.assertIsNone(cache.get('a'))
        self.assertEqual(cache.get('b'), 2)
        self.assertEqual(cache.get('c'), 3)
        self.assertEqual(len(cache), 2)

    def test_update_existing_key_does_not_grow(self):
        cache = LRUCache(2)
        cache.put('a', 1)
        cache.put('b', 2)
        cache.put('a', 10)
        self.assertEqual(len(cache), 2)
        self.assertEqual(cache.get('a'), 10)

    def test_update_marks_most_recently_used(self):
        cache = LRUCache(2)
        cache.put('a', 1)
        cache.put('b', 2)
        cache.put('a', 10)
        cache.put('c', 3)
        self.assertEqual(cache.get('a'), 10)
        self.assertIsNone(cache.get('b'))
        self.assertEqual(cache.get('c'), 3)

    def test_capacity_one(self):
        cache = LRUCache(1)
        cache.put('a', 1)
        cache.put('b', 2)
        self.assertEqual(len(cache), 1)
        self.assertIsNone(cache.get('a'))
        self.assertEqual(cache.get('b'), 2)

    def test_eviction_order_sequence(self):
        cache = LRUCache(3)
        for key, value in [('a', 1), ('b', 2), ('c', 3)]:
            cache.put(key, value)
        cache.get('a')
        cache.put('d', 4)
        self.assertIsNone(cache.get('b'))
        self.assertEqual(cache.get('a'), 1)
        self.assertEqual(cache.get('c'), 3)
        self.assertEqual(cache.get('d'), 4)
        self.assertEqual(len(cache), 3)

    def test_invalid_capacity_zero(self):
        with self.assertRaises(ValueError):
            LRUCache(0)

    def test_invalid_capacity_negative(self):
        with self.assertRaises(ValueError):
            LRUCache(-1)

    def test_invalid_capacity_non_integer(self):
        for bad in (1.5, '2', None, [1]):
            with self.assertRaises(ValueError):
                LRUCache(bad)

    def test_invalid_capacity_bool(self):
        with self.assertRaises(ValueError):
            LRUCache(True)

    def test_values_can_be_none_or_falsy(self):
        cache = LRUCache(2)
        cache.put('a', None)
        self.assertIsNone(cache.get('a', 'sentinel'))
        self.assertEqual(len(cache), 1)
        cache.put('b', 0)
        self.assertEqual(cache.get('b'), 0)
        self.assertEqual(len(cache), 2)


if __name__ == '__main__':
    unittest.main()
