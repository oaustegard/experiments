static void tr_ochar(struct tr_args *a) { if (a->arg[2]) cdef_add(a->arg[0], a->arg[1], a->arg[2]); else cdef_remove(a->arg[0], a->arg[1]); }
