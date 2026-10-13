static void tr_fzoom(struct tr_args *a) { char *s = a->arg[0]; if (s) { void *f = dev_font(dev_pos(s)); if (f) font_setzoom(f, a->arg[1] ? eval(a->arg[1], 0) : 0); } }
