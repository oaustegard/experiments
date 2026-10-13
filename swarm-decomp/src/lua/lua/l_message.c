static void l_message (const char *pname, const char *msg) {
  if (pname) {
    fprintf(stderr, "%s: ", pname);
    fflush(stderr);
  }
  fprintf(stderr, "%s\n", msg);
  fflush(stderr);
}
