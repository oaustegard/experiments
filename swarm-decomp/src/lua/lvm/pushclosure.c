/* pushclosure(L, p, encup, base, ra): create a closure for p (lvm.c) */
static void pushclosure (lua_State *L, Proto *p, UpVal **encup, StkId base, StkId ra) {
  int nup = p->sizeupvalues;
  Upvaldesc *uv = p->upvalues;
  int i;
  LClosure *ncl = luaF_newLclosure(L, nup);
  ncl->p = p;
  ra->value_.gc = (GCObject *)ncl;
  ra->tt_ = LUA_VLCL | BIT_ISCOLLECTABLE;
  for (i = 0; i < nup; i++) {
    if (uv[i].instack)
      ncl->upvals[i] = luaF_findupval(L, base + uv[i].idx);
    else
      ncl->upvals[i] = encup[uv[i].idx];
    luaC_objbarrier(L, ncl, ncl->upvals[i]);
  }
}
