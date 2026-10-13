#include <stdio.h>
#include <string.h>
#include "fn_charread.h"
#include "fn_dev_pos.h"
#include "fn_xmalloc.h"

static void cdef_add(char *dev, char *s, char *val) {
  int pos = -1;
  int n;
  long i;
  char buf[32];

  if (!val)
    return;
  if (charread(&s, buf) < 0)
    return;
  if (dev)
    pos = dev_pos(dev);
  n = cdef_n;
  if (n > 0) {
    for (i = 0; i < n; i++)
      if ((cdef_fn[i] == pos || cdef_fn[i] == 0) && strcmp(cdef_src[i], buf) == 0)
        goto found;
    for (i = 0; i < n && cdef_dst[i]; i++)
      ;
  } else {
    if (n < 0)
      return;
    i = 0;
  }
  if (i == n) {
    if (n == 128)
      return;
    n++;
    cdef_n = n;
  }
found:
  snprintf(cdef_src[i], 32, "%s", buf);
  cdef_dst[i] = xmalloc(strlen(val) + 1);
  strcpy(cdef_dst[i], val);
  cdef_fn[i] = dev ? dev_pos(dev) : 0;
}
