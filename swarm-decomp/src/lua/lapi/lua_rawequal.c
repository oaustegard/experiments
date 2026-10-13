#include "lstate.h"
#include "fn_luaV_equalobj.h"
#include "lua.h"

static TValue *index2value (lua_State *L, int idx);

#define ttisnil(o) (((o)->tt_ & 0xf) == 0)
#define isvalid(L, o) (!ttisnil(o) || (o) != &(L)->l_G->nilvalue)

int lua_rawequal (lua_State *L, int index1, int index2) {
  const TValue *o1 = index2value(L, index1);
  const TValue *o2 = index2value(L, index2);
  return (isvalid(L, o1) && isvalid(L, o2)) ? luaV_equalobj(NULL, o1, o2) : 0;
}
