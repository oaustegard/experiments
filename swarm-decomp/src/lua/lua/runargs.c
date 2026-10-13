static int runargs (lua_State *L, char **argv, int n) {
  int i;
  for (i = 1; i < n; i++) {
    int option = argv[i][1];
    switch (option) {
      case 'e': case 'l': {
        int status;
        char *extra = argv[i] + 2;
        if (*extra == '\0') extra = argv[++i];
        if (option == 'e')
          status = dostring(L, extra, "=(command line)");
        else
          status = dolibrary(L, extra);
        if (status != LUA_OK) return 0;
        break;
      }
      case 'W':
        lua_warning(L, "@on", 0);
        break;
    }
  }
  return 1;
}
