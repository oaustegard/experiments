#include "dumpSize.h"
#define DIBS ((sizeof(size_t) * 8 / 7) + 1)
static void dumpSize (DumpState *D, size_t x) {
  lu_byte buff[DIBS];
  int n = 0;
  do {
    buff[DIBS - (++n)] = x & 0x7f;
    x >>= 7;
  } while (x != 0);
  buff[DIBS - 1] |= 0x80;
  dumpBlock(D, buff + DIBS - n, n);
}
