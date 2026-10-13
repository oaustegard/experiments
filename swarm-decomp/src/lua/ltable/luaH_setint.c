void luaH_setint(lua_State *L, Table *t, lua_Integer key, TValue *value) {
  const TValue *p = luaH_getint(t, key);
  if (p->tt_ == LUA_VABSTKEY) {
    TValue k;
    k.value_.i = key;
    k.tt_ = LUA_VNUMINT;
    luaH_newkey(L, t, &k, value);
  }
  else {
    TValue *q = (TValue *)p;
    q->value_ = value->value_;
    q->tt_ = value->tt_;
  }
}
