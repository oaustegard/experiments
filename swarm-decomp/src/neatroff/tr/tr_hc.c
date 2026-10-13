static void tr_hc(struct tr_args *a) { char *s = a->arg[0]; if (!s || charread(&s, env_hc()) < 0) { char *p = env_hc(); p[0] = 0x5c; p[1] = 0x25; p[2] = 0; } }
