static int dolibrary (lua_State *L, char *globname) {
  int status;
  char *suffix = NULL;
  char *modname = strchr(globname, '=');
  if (modname == NULL) {
    modname = globname;
    suffix = strchr(modname, '-');
  }
  else {
    *modname = '\0';
    modname++;
  }
  lua_getglobal(L, "require");
  lua_pushstring(L, modname);
  status = docall(L, 1, 1);
  if (status == LUA_OK) {
    if (suffix != NULL)
      *suffix = '\0';
    lua_setglobal(L, globname);
  }
  return report(L, status);
}
