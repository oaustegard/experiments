static void tr_pl(struct tr_args *a) { int *p = nreg(0x70); int v = eval_re(a->arg[0] ? a->arg[0] : "11i", *p, 0x76); if (v < 0) v = 0; *nreg(0x70) = v; }
