static void tr_rchar(struct tr_args *a) { char *s = a->arg[0]; if (s) { char **p = &a->arg[1]; do { cdef_remove(0, s); s = *p++; } while (s); } }
