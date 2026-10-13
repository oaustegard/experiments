#include "lstate.h"
#include "Proto.h"
#include "fn_luaC_newobj.h"
#include <string.h>

Proto *luaF_newproto (lua_State *L) {
  GCObject *o = luaC_newobj(L, 10, sizeof(Proto));
  Proto *f = (Proto *)o;
  f->k = NULL;
  f->sizek = 0;
  f->p = NULL;
  f->sizep = 0;
  f->code = NULL;
  f->sizecode = 0;
  f->lineinfo = NULL;
  f->sizelineinfo = 0;
  f->abslineinfo = NULL;
  f->sizeabslineinfo = 0;
  f->upvalues = NULL;
  f->sizeupvalues = 0;
  f->numparams = 0;
  f->is_vararg = 0;
  f->maxstacksize = 0;
  f->locvars = NULL;
  f->sizelocvars = 0;
  f->linedefined = 0;
  f->lastlinedefined = 0;
  f->source = NULL;
  return f;
}
