static int render_rec(int lim)
{
	int olddiv = ren_div;
	int c;
	int nl = 0;
	char s[128];
	struct wb *w;
	void *f;

	ren_div = 0;
	if (ren_level > 99)
		errdie("neatroff: trap recursion depth limit reached\n");
	for (;;) {
		if (lim > ren_level)
			break;
		while (ren_un == 0 && !tr_nextreq()) {
			if (ren_level < lim)
				goto out;
		}
		if (lim > ren_level)
			break;
		if (ren_aborted)
			return ren_aborted;
		c = ren_next();
		if (c < 0) {
			if (bp_final > 1)
				break;
			if (bp_final == 1) {
				bp_final = 2;
				ren_ejectpage(1);
			} else {
				bp_final = 1;
				if (trap_em >= 0)
					trap_exec(trap_em);
			}
			nl = 0;
			f = env_fmt();
			while (fmt_morelines(f))
				ren_passline(f);
			if (ren_nl)
				*nreg(map(".b0")) = fmt_wid(env_fmt()) + wb_wid(env_wb());
			continue;
		}
		ren_partial = c != '\n';
		if (c == ' ' || c == '\n') {
			if (wb_part(env_wb())) {
				nl = c == '\n';
			} else {
				w = env_wb();
				while (fmt_word(env_fmt(), w)) {
					f = env_fmt();
					while (fmt_morelines(f))
						ren_passline(f);
				}
				wb_reset(w);
				if (c == '\n') {
					while (fmt_newline(env_fmt())) {
						f = env_fmt();
						while (fmt_morelines(f))
							ren_passline(f);
					}
				}
				if (*nreg('u') == 0 || *nreg(map(".na")) != 0 ||
				    *nreg(map(".ce")) != 0 || !(*nreg('j') & 0x20)) {
					f = env_fmt();
					while (fmt_fill(f, 0)) {
						while (fmt_morelines(f))
							ren_passline(f);
					}
					while (fmt_morelines(f))
						ren_passline(f);
				}
				if (c == ' ') {
					fmt_space(env_fmt());
					nl = 0;
				} else {
					nl = c == '\n';
				}
			}
			f = env_fmt();
			while (fmt_morelines(f))
				ren_passline(f);
			if (c == '\n') {
				*nreg(map(".b0")) = fmt_wid(env_fmt()) + wb_wid(env_wb());
				if (*nreg(map(".it"))) {
					if (--*nreg(map(".itn")) == 0)
						trap_exec(*nreg(map(".it")));
				}
				if (!wb_part(env_wb())) {
					if (*nreg(map(".ce")) > 1)
						*nreg(map(".ce")) -= 1;
					else
						*nreg(map(".ce")) = 0;
				}
			} else {
				if (ren_nl)
					*nreg(map(".b0")) = fmt_wid(env_fmt()) + wb_wid(env_wb());
				if (c == ' ') {
					ren_nl = nl;
					continue;
				}
				continue;
			}
		} else if (ren_nl) {
			*nreg(map(".b0")) = fmt_wid(env_fmt()) + wb_wid(env_wb());
		}
		ren_back(c);
		w = env_wb();
		if (charnext(s, ren_next, ren_back) >= 0)
			ren_put(w, s, ren_next, ren_back);
		ren_nl = nl;
	}
out:
	ren_div = olddiv;
	return ren_aborted;
}
