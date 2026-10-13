const TValue *luaH_getint(Table *t, lua_Integer key) {
  lua_Unsigned alimit = t->alimit;
  if ((lua_Unsigned)key - 1u < alimit)
    return &t->array[key - 1];
  else if ((t->flags & (1 << 7)) &&
           (((lua_Unsigned)key - 1u) & ~(alimit - 1u)) < alimit) {
    t->alimit = (unsigned int)key;
    return &t->array[key - 1];
  }
  else {
    Node *n = hashint(t, key);
    for (;;) {
      if (n->u.key_tt == LUA_VNUMINT && n->u.key_val.i == key)
        return &n->i_val;
      else {
        int nx = n->u.next;
        if (nx == 0)
          break;
        n += nx;
      }
    }
    return &absentkey;
  }
}
