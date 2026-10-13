#include "lundump.h"

static void loadProtos (LoadState *S, Proto *f) {
  int i;
  int n = loadInt(S);
  f->p = luaM_malloc_(S->L, n * sizeof(Proto *), 0);
  f->sizep = n;
  for (i = 0; i < n; i++)
    f->p[i] = NULL;
  for (i = 0; i < n; i++) {
    f->p[i] = luaF_newproto(S->L);
    if (isblack(f) && iswhite(f->p[i]))
      luaC_barrier_(S->L, (GCObject *)f, (GCObject *)f->p[i]);
    loadFunction(S, f->p[i], f->source);
  }
}
