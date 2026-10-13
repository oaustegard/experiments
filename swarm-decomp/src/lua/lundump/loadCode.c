#include "lundump.h"

static void loadCode (LoadState *S, Proto *f) {
  int n = loadInt(S);
  f->code = luaM_malloc_(S->L, n * sizeof(Instruction), 0);
  f->sizecode = n;
  loadVector(S, f->code, n);
}
