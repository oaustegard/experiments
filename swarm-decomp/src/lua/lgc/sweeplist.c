static inline GCObject **sweeplist(lua_State *L, GCObject **p, int countin) {
  global_State *g = L->l_G;
  int ow = g->currentwhite ^ WHITEBITS;
  int white = g->currentwhite & WHITEBITS;
  GCObject *curr;
  while ((curr = *p) != NULL && countin-- > 0) {
    if (curr->marked & ow) {
      *p = curr->next;
      freeobj(L, curr);
    }
    else {
      curr->marked = cast_byte((curr->marked & ~(maskcolors | AGEBITS)) | white);
      p = &curr->next;
    }
  }
  return (*p == NULL) ? NULL : p;
}
