static void packint (luaL_Buffer *b, lua_Unsigned n, int islittle, int size, int neg) {
  char *buff = luaL_prepbuffsize(b, size);
  int i;
  buff[islittle ? 0 : size - 1] = (char)(n & MC);
  for (i = 1; i < size; i++) {
    n >>= NB;
    buff[islittle ? i : size - 1 - i] = (char)(n & MC);
  }
  if (neg && size > SZINT) {
    for (i = SZINT; i < size; i++)
      buff[islittle ? i : size - 1 - i] = (char)MC;
  }
  luaL_addsize(b, size);
}
