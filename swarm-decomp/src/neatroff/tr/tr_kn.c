static void tr_kn(struct tr_args *a) { char *s = a->arg[0]; if (s) { int *p = nreg(map(".kn")); *p = eval(s, 0); } }
