static void tr_nn(struct tr_args *a) { int n = 1; if (a->arg[0]) n = eval(a->arg[0], 0); int *p = nreg(map(".nn")); *p = n; }
