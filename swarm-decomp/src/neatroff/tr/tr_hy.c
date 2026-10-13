static void tr_hy(struct tr_args *a) { int v = 1; if (a->arg[0]) { int *p = nreg(map(".hy")); v = eval_re(a->arg[0], *p, 0); } *nreg(map(".hy")) = v; }
