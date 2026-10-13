static void tr_char(struct tr_args *a) { if (a->arg[1]) cdef_add(0, a->arg[0], a->arg[1]); else cdef_remove(0, a->arg[0]); }
