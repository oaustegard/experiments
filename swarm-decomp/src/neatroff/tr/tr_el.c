static void tr_el(struct tr_args *a) { int d = 1; if (ie_depth > 0) { ie_depth--; d = ie_cond[ie_depth]; } cp_blk(d); }
