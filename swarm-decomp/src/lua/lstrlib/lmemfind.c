static const char *lmemfind (const char *s1, size_t l1, const char *s2, size_t l2) {
  if (l2 == 0) return s1;
  else if (l2 > l1) return NULL;
  else {
    const char *init;
    l2--;
    l1 = l1-l2;
    while (l1 > 0 && (init = (const char *)memchr(s1, *s2, l1)) != NULL) {
      init++;
      if (memcmp(init, s2+1, l2) == 0)
        return init-1;
      else {
        l1 -= init-s1;
        s1 = init;
      }
    }
    return NULL;
  }
}
