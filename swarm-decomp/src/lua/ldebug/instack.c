static int instack (CallInfo *ci, const TValue *o) {
  int pos;
  StkId base = ci->func + 1;
  for (pos = 0; base + pos < ci->top; pos++) {
    if (o == base + pos)
      return pos;
  }
  return -1;
}
