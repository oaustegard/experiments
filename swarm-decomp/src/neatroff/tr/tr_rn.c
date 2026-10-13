static void tr_rn(struct tr_args *a) { if (a->arg[1]) { int b = map(a->arg[1]); str_rn(map(a->arg[0]), b); } }
