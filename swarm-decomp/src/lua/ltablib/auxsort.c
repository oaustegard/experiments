static void auxsort (lua_State *L, IdxT lo, IdxT up, unsigned int rnd) {
  IdxT p, n;
  while (lo < up) {
    lua_geti(L, 1, lo);
    lua_geti(L, 1, up);
    if (sort_comp(L, -1, -2))
      set2(L, lo, up);
    else
      lua_settop(L, -3);
    if (up - lo == 1)
      break;
    if (up - lo < 100u || rnd == 0)
      p = (lo + up)/2;
    else
      p = choosePivot(lo, up, rnd);
    lua_geti(L, 1, p);
    lua_geti(L, 1, lo);
    if (sort_comp(L, -2, -1))
      set2(L, p, lo);
    else {
      lua_settop(L, -2);
      lua_geti(L, 1, up);
      if (sort_comp(L, -1, -2))
        set2(L, p, up);
      else
        lua_settop(L, -3);
    }
    if (up - lo == 2)
      break;
    lua_geti(L, 1, p);
    lua_pushvalue(L, -1);
    lua_geti(L, 1, up - 1);
    set2(L, p, up - 1);
    p = partition(L, lo, up);
    if (p - lo < up - p) {
      auxsort(L, lo, p - 1, rnd);
      n = p - lo;
      lo = p + 1;
    }
    else {
      auxsort(L, p + 1, up, rnd);
      n = up - p;
      up = p - 1;
    }
    if ((up - lo) / 128 > n)
      rnd = l_randomizePivot();
  }
}
