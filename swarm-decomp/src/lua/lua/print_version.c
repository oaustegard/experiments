static void print_version (void) {
  fwrite(LUA_COPYRIGHT, sizeof(char), strlen(LUA_COPYRIGHT), stdout);
  fwrite("\n", sizeof(char), 1, stdout);
  fflush(stdout);
}
