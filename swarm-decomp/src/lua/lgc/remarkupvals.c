static int remarkupvals(global_State *g) {
  lua_State *thread;
  lua_State **p = &g->twups;
  int work = 0;
  while ((thread = *p) != NULL) {
    work++;
    if (!iswhite(thread) && thread->openupval != NULL)
      p = &thread->twups;
    else {
      UpVal *uv;
      *p = thread->twups;
      thread->twups = thread;
      for (uv = (UpVal *)thread->openupval; uv != NULL; uv = uv->u.open.next) {
        work++;
        if (!iswhite(uv))
          markvalue(g, uv->v.p);
      }
    }
  }
  return work;
}
