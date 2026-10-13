void luaK_patchlist (FuncState *fs, int list, int target) {
  patchlistaux(fs, list, target, NO_REG, target);
}
