#include "fn_luaM_free_.h"
#include "fn_luaF_freeproto.h"

void luaF_freeproto (lua_State *L, Proto *f) {
  luaM_free_(L, f->code, f->sizecode * sizeof(Instruction));
  luaM_free_(L, f->p, f->sizep * sizeof(Proto *));
  luaM_free_(L, f->k, f->sizek * sizeof(TValue));
  luaM_free_(L, f->lineinfo, f->sizelineinfo * sizeof(ls_byte));
  luaM_free_(L, f->abslineinfo, f->sizeabslineinfo * sizeof(AbsLineInfo));
  luaM_free_(L, f->locvars, f->sizelocvars * sizeof(LocVar));
  luaM_free_(L, f->upvalues, f->sizeupvalues * sizeof(Upvaldesc));
  luaM_free_(L, f, sizeof(Proto));
}
