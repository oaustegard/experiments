static const char *getupvalname (CallInfo *ci, const TValue *o, const char **name) {
  LClosure *c = (LClosure *)ci->func->value_.gc;
  int i;
  for (i = 0; i < c->nupvalues; i++) {
    if (c->upvals[i]->v.p == o) {
      *name = upvalname(c->p, i);
      return "upvalue";
    }
  }
  return NULL;
}
