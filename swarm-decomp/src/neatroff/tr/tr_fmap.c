static void tr_fmap(struct tr_args *a) { char *s = a->arg[0]; if (s) { void *f = dev_font(dev_pos(s)); if (f && a->arg[1]) font_map(f, a->arg[1], a->arg[2]); } }
