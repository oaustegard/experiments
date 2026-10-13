static const char *varinfo (lua_State *L, const TValue *o) {
  CallInfo *ci = L->ci;
  const char *name = NULL;
  const char *kind = NULL;
  if (!(ci->callstatus & 2)) {
    kind = getupvalname(ci, o, &name);
    if (!kind) {
      int reg = instack(ci, o);
      if (reg >= 0)
        kind = getobjname(((LClosure *)ci->func->value_.gc)->p, (int)(ci->u.l.savedpc - ((LClosure *)ci->func->value_.gc)->p->code) - 1, reg, &name);
    }
  }
  return formatvarinfo(L, kind, name);
}
