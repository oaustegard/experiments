#include "lstate.h"
#include "LClosure.h"
#include "Zio.h"
#include "fn_luaD_protectedparser.h"
#include "fn_luaZ_init.h"
#include "lua.h"

LUA_API int lua_load (lua_State *L, lua_Reader reader, void *data,
                      const char *chunkname, const char *mode) {
  ZIO z;
  int status;
  if (!chunkname) chunkname = "?";
  luaZ_init(L, &z, reader, data);
  status = luaD_protectedparser(L, &z, chunkname, mode);
  if (status == 0) {
    LClosure *f = (LClosure *)(L->top - 1)->value_.gc;
    if (f->nupvalues >= 1) {
      const TValue *gt = &((Table *)L->l_G->l_registry.value_.gc)->array[2 - 1];
      TValue *uv = (TValue *)f->upvals[0]->v.p;
      uv->value_ = gt->value_;
      uv->tt_ = gt->tt_;
      if ((gt->tt_ & 0x40) && (f->upvals[0]->marked & 0x20) && (gt->value_.gc->marked & 0x18))
        luaC_barrier_(L, (GCObject *)f->upvals[0], gt->value_.gc);
    }
  }
  return status;
}
