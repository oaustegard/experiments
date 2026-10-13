#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "fn_charread.h"
#include "fn_dev_pos.h"
#include "fn_free.h"

static void cdef_remove(char *dev, char *s) {
  char buf[32];
  int pos = -1;
  int n;
  long i;

  if (dev)
    pos = dev_pos(dev);
  if (s) {
    if (charread(&s, buf) < 0)
      return;
    n = cdef_n;
    if (n > 0)
      for (i = 0; i < n; i++) {
        if (strcmp(cdef_src[i], buf) == 0 && (!dev || (pos > 0 && cdef_fn[i] == pos))) {
          free(cdef_dst[i]);
          cdef_src[i][0] = 0;
          cdef_dst[i] = 0;
        }
      }
  }
}
