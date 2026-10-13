#include "lundump.h"

static lu_byte loadByte (LoadState *S) {
  ZIO *z = S->Z;
  int b = (z->n-- > 0) ? (unsigned char)*z->p++ : luaZ_fill(z);
  if (b == -1)
    error(S, "truncated chunk");
  return (lu_byte)b;
}
