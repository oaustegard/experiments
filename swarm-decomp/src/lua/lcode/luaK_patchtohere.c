void luaK_patchtohere (FuncState *fs, int list) {
  int hr = fs->pc;
  luaK_getlabel(fs);
  patchlistaux(fs, list, hr, NO_REG, hr);
}
