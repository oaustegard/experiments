static void tr_cp(struct tr_args *a) { char *s = a->arg[0]; if (s) { int *p = nreg(0x43); *p = (int)strtol(s, 0, 10); } }
