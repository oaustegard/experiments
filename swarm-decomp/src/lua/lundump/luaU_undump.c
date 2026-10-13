#include "lundump.h"

LClosure *luaU_undump (lua_State *L, ZIO *Z, const char *name) {
  LoadState S;
  LClosure *cl;
  if (*name == '@' || *name == '=')
    S.name = name + 1;
  else if (*name == 0x1b)  /* LUA_SIGNATURE[0] */
    S.name = "binary string";
  else
    S.name = name;
  S.L = L;
  S.Z = Z;
  checkHeader(&S);
  cl = luaF_newLclosure(L, loadByte(&S));
  L->top->value_.gc = (GCObject *)cl;
  L->top->tt_ = 0x46;  /* LUA_VLCL | BIT_ISCOLLECTABLE */
  luaD_inctop(L);
  cl->p = luaF_newproto(L);
  if (isblack(cl) && iswhite(cl->p))
    luaC_barrier_(L, (GCObject *)cl, (GCObject *)cl->p);
  loadFunction(&S, cl->p, NULL);
  return cl;
}
