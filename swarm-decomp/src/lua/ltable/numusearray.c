#include "Table.h"

static unsigned int numusearray (const Table *t, unsigned int *nums) {
  unsigned int lg;
  unsigned int ttlg;
  unsigned int ause = 0;
  unsigned int i = 1;
  unsigned int asize = t->alimit;
  for (lg = 0, ttlg = 1; lg <= 31; lg++, ttlg *= 2) {
    unsigned int lc = 0;
    unsigned int lim = ttlg;
    if (lim > asize) {
      lim = asize;
      if (i > lim)
        break;
    }
    for (; i <= lim; i++) {
      if ((t->array[i - 1].tt_ & 0xf) != 0)
        lc++;
    }
    nums[lg] += lc;
    ause += lc;
  }
  return ause;
}
