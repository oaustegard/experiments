#include <lua.h>
#include <lobject.h>
#include <fn_luaV_shiftl.h>

#define VM_SHIFT_NBITS 64

lua_Integer luaV_shiftl (lua_Integer x, lua_Integer y) {
  if (y < 0) {  /* shift right? */
    if (y <= -VM_SHIFT_NBITS) return 0;
    else return (lua_Integer)((lua_Unsigned)x >> (lua_Unsigned)(-y));
  }
  else {  /* shift left */
    if (y >= VM_SHIFT_NBITS) return 0;
    else return (lua_Integer)((lua_Unsigned)x << (lua_Unsigned)y);
  }
}
