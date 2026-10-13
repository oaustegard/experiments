static void tr_lg(struct tr_args *a) { char *s = a->arg[0]; if (s) { int *p = nreg(map(".lg")); *p = eval(s, 0); } }
