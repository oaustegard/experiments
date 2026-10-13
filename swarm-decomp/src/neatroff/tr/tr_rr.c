static void tr_rr(struct tr_args *a) { char *s = a->arg[0]; if (s) { char **p = &a->arg[1]; do { num_del(map(s)); s = *p++; } while (s); } }
