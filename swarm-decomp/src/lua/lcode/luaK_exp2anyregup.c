void luaK_exp2anyregup (FuncState *fs, expdesc *e) {
  if (e->k != VUPVAL || e->t != e->f)
    luaK_exp2anyreg(fs, e);
}
