static inline GCObject **sweeplist_fin(lua_State *L, GCObject **p, int countin) {
  global_State *g = L->l_G;
  int ow = g->currentwhite ^ WHITEBITS;
  GCObject *curr;
  while ((curr = *p) != NULL && countin-- > 0) {
    if (!(curr->marked & ow)) {
      curr->marked = cast_byte((curr->marked & ~(maskcolors | AGEBITS)) | luaC_white(g));
      p = &curr->next;
    }
    else {
      *p = curr->next;
      freeobj(L, curr);
    }
  }
  return (*p == NULL) ? NULL : p;
}
