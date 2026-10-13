#include "Table.h"
#include "TString.h"
#include "lobject.h"

int luaV_flttointeger (lua_Number n, lua_Integer *p, int mode);
const TValue *luaH_getint (Table *t, lua_Integer key);
const TValue *luaH_getshortstr (Table *t, TString *key);

const TValue *luaH_get (Table *t, const TValue *key) {
  switch (key->tt_ & 0x3f) {
    case 0x04:
      return luaH_getshortstr(t, (TString *)key->value_.gc);
    case 0x03:
      return luaH_getint(t, key->value_.i);
    case 0x00:
      return &absentkey;
    case 0x13: {
      lua_Integer k;
      if (luaV_flttointeger(key->value_.n, &k, 0))
        return luaH_getint(t, k);
    }
    /* FALLTHROUGH */
    default:
      return getgeneric(t, key, 0);
  }
}
