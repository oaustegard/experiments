static void separatetobefnz(global_State *g, int all) {
  GCObject *curr;
  GCObject **p = &g->finobj;
  GCObject **lastnext = findlast(&g->tobefnz);
  while ((curr = *p) != g->finobjold1) {
    if (!(iswhite(curr) || all))
      p = &curr->next;
    else {
      if (curr == g->finobjsur)
        g->finobjsur = curr->next;
      *p = curr->next;
      curr->next = *lastnext;
      *lastnext = curr;
      lastnext = &curr->next;
    }
  }
}
