#include <limits.h>
#include "lobject.h"
#include "Table.h"

const TValue *luaH_getint (Table *t, lua_Integer key);

static inline lua_Unsigned hash_search (Table *t, lua_Unsigned j) {
  lua_Unsigned i;
  if (j == 0) j++;
  do {
    i = j;
    if (j <= (lua_Unsigned)LUA_MAXINTEGER / 2)
      j *= 2;
    else {
      j = LUA_MAXINTEGER;
      if ((luaH_getint(t, j)->tt_ & 0xf) == 0)
        break;
      else
        return j;
    }
  } while ((luaH_getint(t, j)->tt_ & 0xf) != 0);
  while (j - i > 1u) {
    lua_Unsigned m = (i + j) / 2;
    if ((luaH_getint(t, m)->tt_ & 0xf) == 0)
      j = m;
    else
      i = m;
  }
  return i;
}
