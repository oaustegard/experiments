static void tr_ffsc(struct tr_args *a) { char *s = a->arg[0]; if (s) { void *f = dev_font(dev_pos(s)); if (f) { font_scrp(f, a->arg[1]); font_lang(f, a->arg[2]); } } }
