void luaK_semerror (LexState *ls, const char *msg) {
  ls->t.token = 0;
  luaX_syntaxerror(ls, msg);
}
