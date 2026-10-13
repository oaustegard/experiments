void luaK_exp2val (FuncState *fs, expdesc *e) {
  if (e->t != e->f)
    luaK_exp2anyreg(fs, e);
  else
    luaK_dischargevars(fs, e);
}
