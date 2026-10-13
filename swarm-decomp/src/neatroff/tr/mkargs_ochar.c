static void mkargs_ochar(struct tr_args *a) { char *s = read_name(0); sbuf_append((struct sbuf *)a, s); sbuf_add((struct sbuf *)a, 0); free(s); mkargs_ds(a); }
