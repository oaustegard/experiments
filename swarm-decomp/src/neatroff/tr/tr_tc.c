static void tr_tc(struct tr_args *a) { char *s = a->arg[0]; if (!s || charread(&s, env_tc()) < 0) *env_tc() = 0; }
