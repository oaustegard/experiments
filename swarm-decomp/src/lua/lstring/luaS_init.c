#include "lstate.h"
#include "TString.h"
#include "fn_luaM_malloc_.h"
#include "fn_luaC_fix.h"
#include "internshrstr.h"

#define MINSTRTABSIZE 128
#define MEMERRMSG "not enough memory"

void luaS_init (lua_State *L) {
  global_State *g = G(L);
  stringtable *tb = &G(L)->strt;
  int i, j;
  tb->hash = (TString **)luaM_malloc_(L, MINSTRTABSIZE * sizeof(TString *), 0);
  for (i = 0; i < MINSTRTABSIZE; i++)
    tb->hash[i] = NULL;
  tb->size = MINSTRTABSIZE;
  g->memerrmsg = internshrstr(L, MEMERRMSG, sizeof(MEMERRMSG) - 1);
  luaC_fix(L, (GCObject *)g->memerrmsg);
  {
    TString **p = &g->strcache[0][0];
    for (i = 0; i < STRCACHE_N * STRCACHE_M; i++)
      p[i] = g->memerrmsg;
  }
}
