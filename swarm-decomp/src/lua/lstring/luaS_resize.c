#include "lstate.h"
#include "TString.h"
#include "fn_luaM_realloc_.h"

static void tablerehash (TString **vect, int osize, int nsize);

void luaS_resize (lua_State *L, int nsize) {
  stringtable *tb = &(L->l_G->strt);
  int osize = tb->size;
  TString **newvect;
  if (nsize < osize)
    tablerehash(tb->hash, osize, nsize);
  newvect = (TString **)luaM_realloc_(L, tb->hash, osize * sizeof(TString *),
                                      nsize * sizeof(TString *));
  if (newvect == NULL) {
    if (nsize < osize)
      tablerehash(tb->hash, nsize, osize);
    return;
  }
  tb->hash = newvect;
  tb->size = nsize;
  if (nsize > osize)
    tablerehash(newvect, osize, nsize);
}
