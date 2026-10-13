static const char *formatvarinfo (lua_State *L, const char *kind, const char *name) {
  if (kind == NULL)
    return "";
  else
    return luaO_pushfstring(L, " (%s '%s')", kind, name);
}
