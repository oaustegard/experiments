static void print_usage (const char *badoption) {
  const char *fmt = "unrecognized option '%s'\n";
  fprintf(stderr, "%s: ", progname);
  fflush(stderr);
  if (badoption[1] == 'e' || badoption[1] == 'l')
    fmt = "'%s' needs argument\n";
  fprintf(stderr, fmt, badoption);
  fprintf(stderr,
    "usage: %s [options] [script [args]]\n"
    "Available options are:\n"
    "  -e stat   execute string 'stat'\n"
    "  -i        enter interactive mode after executing 'script'\n"
    "  -l mod    require library 'mod' into global 'mod'\n"
    "  -l g=mod  require library 'mod' into global 'g'\n"
    "  -v        show version information\n"
    "  -E        ignore environment variables\n"
    "  -W        turn warnings on\n"
    "  --        stop handling options\n"
    "  -         stop handling options and execute stdin\n",
    progname);
  fflush(stderr);
}
