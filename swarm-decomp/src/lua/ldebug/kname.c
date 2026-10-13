static const char *kname (const Proto *p, int index, const char **name) {
  TValue *kvalue = &p->k[index];
  if ((kvalue->tt_ & 0xf) == 4) {
    TString *ts = (TString *)kvalue->value_.gc;
    *name = ts->contents;
    return "constant";
  }
  else {
    *name = "?";
    return NULL;
  }
}
