/* floatforloop(L, ra): float step of a numeric for loop (lvm.c, l_sinline) */
static int floatforloop (lua_State *L, StkId ra) {
  lua_Number step = fltvalue(s2v(ra + 2));
  lua_Number limit = fltvalue(s2v(ra + 1));
  lua_Number idx = fltvalue(s2v(ra));
  idx = idx + step;
  if (0 < step ? idx <= limit : limit <= idx) {
    ra->value_.n = idx;
    (ra + 3)->value_.n = idx;
    (ra + 3)->tt_ = LUA_VNUMFLT;
    return 1;
  }
  else
    return 0;
}
