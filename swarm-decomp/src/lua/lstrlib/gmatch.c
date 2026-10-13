static int gmatch (lua_State *L) {
  size_t ls, lp;
  const char *s = luaL_checklstring(L, 1, &ls);
  const char *p = luaL_checklstring(L, 2, &lp);
  size_t init = posrelatI(luaL_optinteger(L, 3, 1), ls) - 1;
  GMatchState *gm;
  lua_settop(L, 2);
  gm = (GMatchState *)lua_newuserdatauv(L, sizeof(GMatchState), 0);
  if (init > ls)
    init = ls + 1;
  gm->ms.L = L;
  gm->ms.matchdepth = 200;
  gm->ms.src_init = s;
  gm->ms.src_end = s + ls;
  gm->ms.p_end = p + lp;
  gm->src = s + init;
  gm->p = p;
  gm->lastmatch = NULL;
  lua_pushcclosure(L, gmatch_aux, 3);
  return 1;
}
