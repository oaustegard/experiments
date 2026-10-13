static GCObject **findlast(GCObject **p) {
  while (*p != NULL)
    p = &(*p)->next;
  return p;
}
