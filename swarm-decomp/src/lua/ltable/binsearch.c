#include "Table.h"

static inline unsigned int binsearch (const TValue *array, unsigned int i, unsigned int j) {
  while (j - i > 1u) {
    unsigned int m = (i + j) / 2;
    if ((array[m - 1].tt_ & 0xf) == 0)
      j = m;
    else
      i = m;
  }
  return i;
}
