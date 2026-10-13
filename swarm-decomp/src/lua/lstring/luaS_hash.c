#include "lstate.h"

unsigned int luaS_hash (const char *str, size_t l, unsigned int seed) {
  unsigned int h = seed ^ (unsigned int)l;
  for (; l > 0; l--)
    h ^= ((h << 5) + (h >> 2) + (unsigned char)str[l - 1]);
  return h;
}
