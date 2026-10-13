#include "lstate.h"
#include "Table.h"



void luaH_finishset (lua_State *L, Table *t, const TValue *key, const TValue *slot, TValue *value) {
  if (slot->tt_ == 0x20)
    luaH_newkey(L, t, key, value);
  else {
    TValue *s = (TValue *)slot;
    s->value_ = value->value_;
    s->tt_ = value->tt_;
  }
}
