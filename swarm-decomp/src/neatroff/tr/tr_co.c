static void tr_co(struct tr_args *a) { char *s1 = a->arg[0]; char *s2 = a->arg[1]; if (s1 && s2 && str_get(map(s1))) { char *v = str_get(map(s1)); str_set(map(s2), v); } }
