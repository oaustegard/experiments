static const char *funcnamefromcall (lua_State *L, CallInfo *ci, const char **name) {
  if (ci->callstatus & 8) {
    *name = "?";
    return "hook";
  }
  else if (ci->callstatus & 0x80) {
    *name = "__gc";
    return "metamethod";
  }
  else if (!(ci->callstatus & 2))
    return funcnamefromcode(L, ((LClosure *)ci->func->value_.gc)->p, (int)(ci->u.l.savedpc - ((LClosure *)ci->func->value_.gc)->p->code) - 1, name);
  else return NULL;
}
