void luaH_set(lua_State *L, Table *t, const TValue *key, TValue *value) {
  const TValue *slot = luaH_get(t, key);
  if (slot->tt_ == LUA_VABSTKEY)
    luaH_newkey(L, t, key, value);
  else {
    TValue *p = (TValue *)slot;
    p->value_ = value->value_;
    p->tt_ = value->tt_;
  }
}
