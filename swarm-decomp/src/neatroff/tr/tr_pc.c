static void tr_pc(struct tr_args *a) { char *s = a->arg[0]; if (!s || charread(&s, c_pc) < 0) c_pc[0] = 0; }
