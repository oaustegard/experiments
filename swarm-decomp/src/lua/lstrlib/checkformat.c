static void checkformat (lua_State *L, const char *form, const char *flags, int precision) {
  const char *spec = form + 1;
  spec += strspn(spec, flags);
  if (*spec != '0') {
    spec = get2digits(spec);
    if (*spec == '.' && precision) {
      spec++;
      spec = get2digits(spec);
    }
  }
  if (!isalpha(uchar(*spec)))
    luaL_error(L, "invalid conversion specification: '%s'", form);
}
