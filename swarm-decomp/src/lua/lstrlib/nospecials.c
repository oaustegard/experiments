static int nospecials (const char *p, size_t l) {
  size_t upto = 0;
  do {
    if (strpbrk(p + upto, "^$*+?.([%-"))
      return 0;
    upto += strlen(p + upto) + 1;
  } while (upto <= l);
  return 1;
}
