static int gmatch_aux (lua_State *L) {
  GMatchState *gm = (GMatchState *)lua_touserdata(L, lua_upvalueindex(3));
  const char *src;
  gm->ms.L = L;
  for (src = gm->src; src <= gm->ms.src_end; src++) {
    const char *e;
    gm->ms.level = 0;
    if ((e = match(&gm->ms, src, gm->p)) != NULL && e != gm->lastmatch) {
      int i;
      unsigned char nlevels;
      gm->src = gm->lastmatch = e;
      nlevels = gm->ms.level;
      if (nlevels == 0 && src)
        nlevels = 1;
      luaL_checkstack(gm->ms.L, nlevels, "too many captures");
      for (i = 0; i < nlevels; i++)
        push_onecapture(&gm->ms, i, src, e);
      return nlevels;
    }
  }
  return 0;
}
