static const char *upvalname (const Proto *p, int uv) {
  TString *s = p->upvalues[uv].name;
  if (s == NULL) return "?";
  else return s->contents;
}
