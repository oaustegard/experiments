static const char *isEnv (const Proto *p, int pc, Instruction i, int isup) {
  int t = DGETARG_B(i);
  const char *name;
  if (isup) {
    if (p->upvalues[t].name == NULL)
      return "field";
    name = upvalname(p, t);
  }
  else
    basicgetobjname(p, &pc, t, &name);
  return (name && strcmp(name, "_ENV") == 0) ? "global" : "field";
}
