static void tr_rm(struct tr_args *a) { char *s = a->arg[0]; if (s) { char **p = &a->arg[1]; do { str_rm(map(s)); s = *p++; } while (s); } }
