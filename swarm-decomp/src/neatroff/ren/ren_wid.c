int ren_wid(int (*next)(void), void (*back)(int))
{
	struct wb w;
	char buf1[32];
	char buf2[128];
	int c;
	int wid;

	wb_init(&w);
	charnext(buf1, next, back);
	odiv_beg();
	c = next();
	if (c != '\n' && c >= 0) {
		for (;;) {
			back(c);
			if (charnext(buf2, next, back) < 0)
				break;
			if (strcmp(buf1, buf2) == 0)
				break;
			ren_put(&w, buf2, next, back);
			c = next();
			if (c < 0 || c == '\n')
				break;
		}
	}
	odiv_end();
	wid = wb_wid(&w);
	wb_wconf(&w, nreg(map("ct")), nreg(map("st")), nreg(map("sb")), nreg(map("bbllx")),
		nreg(map("bblly")), nreg(map("bburx")), nreg(map("bbury")));
	wb_done(&w);
	return wid;
}
