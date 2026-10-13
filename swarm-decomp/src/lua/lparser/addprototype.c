#include "FuncState.h"
#include "fn_luaF_newproto.h"
#include "fn_luaM_growaux_.h"
#include "luaC_objbarrier.h"

static Proto *addprototype (LexState *ls) {
  Proto *clp;
  lua_State *L = ls->L;
  FuncState *fs = ls->fs;
  Proto *f = fs->f;
  if (fs->np >= f->sizep) {
    int oldsize = f->sizep;
    f->p = (Proto **)luaM_growaux_(L, f->p, fs->np, &f->sizep, sizeof(Proto *), 0x1ffff, "functions");
    if (oldsize < f->sizep) {
      do {
        f->p[oldsize++] = NULL;
      } while (oldsize < f->sizep);
    }
  }
  f->p[fs->np++] = clp = luaF_newproto(L);
  luaC_objbarrier(L, f, clp);
  return clp;
}
