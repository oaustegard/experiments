static int docall (lua_State *L, int narg, int nres) {
  int status;
  int base = lua_gettop(L) - narg;
  lua_pushcfunction(L, msghandler);
  lua_insert(L, base);
  globalL = L;
  {
    struct sigaction sa;
    sa.sa_handler = laction;
    sa.sa_flags = 0;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGINT, &sa, NULL);
  }
  status = lua_pcall(L, narg, nres, base);
  {
    struct sigaction sa;
    sa.sa_handler = SIG_DFL;
    sa.sa_flags = 0;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGINT, &sa, NULL);
  }
  lua_remove(L, base);
  return status;
}
