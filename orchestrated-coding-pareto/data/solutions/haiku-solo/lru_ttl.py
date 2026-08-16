class TTLCache:
    def __init__(self, capacity: int, ttl: float, clock):
        self.capacity = capacity
        self.ttl = ttl
        self.clock = clock
        self._cache = {}

    def _is_expired(self, insertion_time):
        """Check if an entry is expired."""
        return self.clock() - insertion_time >= self.ttl

    def _count_non_expired(self):
        """Count non-expired entries."""
        return sum(
            1 for value, insertion_time in self._cache.values()
            if not self._is_expired(insertion_time)
        )

    def put(self, key, value) -> None:
        current_time = self.clock()

        # If key exists, update it (refreshes TTL and marks as MRU)
        if key in self._cache:
            del self._cache[key]
            self._cache[key] = (value, current_time)
        else:
            # New key - check if we need to evict
            non_expired_count = self._count_non_expired()
            if non_expired_count >= self.capacity:
                # Remove expired entries
                keys_to_remove = [
                    k for k, (v, t) in self._cache.items()
                    if self._is_expired(t)
                ]
                for k in keys_to_remove:
                    del self._cache[k]

                # If still at capacity, evict LRU non-expired entry
                non_expired_count = self._count_non_expired()
                if non_expired_count >= self.capacity:
                    for k in self._cache:
                        v, t = self._cache[k]
                        if not self._is_expired(t):
                            del self._cache[k]
                            break

            # Add new entry
            self._cache[key] = (value, current_time)

    def get(self, key, default=None):
        if key not in self._cache:
            return default

        value, insertion_time = self._cache[key]

        if self._is_expired(insertion_time):
            # Remove expired entry
            del self._cache[key]
            return default

        # Entry is valid - mark as MRU by moving to end (without updating timestamp)
        del self._cache[key]
        self._cache[key] = (value, insertion_time)
        return value

    def __len__(self) -> int:
        return self._count_non_expired()
