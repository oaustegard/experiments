static void tr_hlm(struct tr_args *a) { int v = 0; if (a->arg[0]) { int *p = nreg(map(".hlm")); v = eval_re(a->arg[0], *p, 0); } *nreg(map(".hlm")) = v; }
