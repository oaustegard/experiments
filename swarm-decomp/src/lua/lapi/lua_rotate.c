#include "lstate.h"
#include "lua.h"

LUA_API void lua_rotate (lua_State *L, int idx, int n) {
  StkId p, t, m;
  StkId from, to;
  t = L->top - 1;
  p = index2stack(L, idx);
  m = (n >= 0 ? t - n : p - n - 1);
  for (from = p, to = m; from < to; from++, to--) {
    TValue temp;
    temp.value_ = from->value_; temp.tt_ = from->tt_;
    from->value_ = to->value_; from->tt_ = to->tt_;
    to->value_ = temp.value_; to->tt_ = temp.tt_;
  }
  for (from = m + 1, to = t; from < to; from++, to--) {
    TValue temp;
    temp.value_ = from->value_; temp.tt_ = from->tt_;
    from->value_ = to->value_; from->tt_ = to->tt_;
    to->value_ = temp.value_; to->tt_ = temp.tt_;
  }
  for (from = p, to = t; from < to; from++, to--) {
    TValue temp;
    temp.value_ = from->value_; temp.tt_ = from->tt_;
    from->value_ = to->value_; from->tt_ = to->tt_;
    to->value_ = temp.value_; to->tt_ = temp.tt_;
  }
}
