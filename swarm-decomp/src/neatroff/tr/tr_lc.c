static void tr_lc(struct tr_args *a) { char *s = a->arg[0]; if (!s || charread(&s, env_lc()) < 0) *env_lc() = 0; }
