#define has_error 1
#define has_i 2
#define has_v 4
#define has_e 8
#define has_E 16
#define has_W 32

static int collectargs (char **argv, int *first) {
  int args = 0;
  int i;
  if (argv[0] != NULL) {
    if (argv[0][0]) progname = argv[0];
  }
  else {
    *first = -1;
    return 0;
  }
  for (i = 1; argv[i] != NULL; i++) {
    *first = i;
    if (argv[i][0] != '-')
      return args;
    switch (argv[i][1]) {
      case '-':
        if (argv[i][2] != '\0') return has_error;
        *first = i + 1;
        return args;
      case '\0':
        return args;
      case 'E':
        if (argv[i][2] != '\0') return has_error;
        args |= has_E;
        break;
      case 'W':
        if (argv[i][2] != '\0') return has_error;
        break;
      case 'i':
        args |= has_i;
      case 'v':
        if (argv[i][2] != '\0') return has_error;
        args |= has_v;
        break;
      case 'e':
        args |= has_e;
      case 'l':
        if (argv[i][2] == '\0') {
          i++;
          if (argv[i] == NULL || argv[i][0] == '-') return has_error;
        }
        break;
      default:
        return has_error;
    }
  }
  *first = 0;
  return args;
}
