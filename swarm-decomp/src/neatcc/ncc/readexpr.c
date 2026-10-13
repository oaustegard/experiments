static void readexpr(void)
{
	char *p;

	readcexpr();
	p = tok_see();
	if (p[0] == '=' && !p[1]) {
		tok_get();
		readexpr();
		doassign();
		return;
	}
	p = tok_see();
	if (p[0] == '+' && p[1] == '=' && !p[2]) {
		tok_get();
		opassign(0x10, p);
		return;
	}
	p = tok_see();
	if (p[0] == '-' && p[1] == '=' && !p[2]) {
		tok_get();
		opassign(0x11, p);
		return;
	}
	p = tok_see();
	if (p[0] == '*' && p[1] == '=' && !p[2]) {
		tok_get();
		opassign(0x40, p);
		return;
	}
	p = tok_see();
	if (p[0] == '/' && p[1] == '=' && !p[2]) {
		tok_get();
		opassign(0x41, p);
		return;
	}
	p = tok_see();
	if (p[0] == '%' && p[1] == '=' && !p[2]) {
		tok_get();
		opassign(0x42, p);
		return;
	}
	p = tok_see();
	if (strcmp("<<=", p) == 0) {
		tok_get();
		opassign(0x20, p);
		return;
	}
	p = tok_see();
	if (strcmp(">>=", p) == 0) {
		tok_get();
		opassign(0x21, p);
		return;
	}
	p = tok_see();
	if (p[0] == '&' && p[1] == '=' && !p[2]) {
		tok_get();
		opassign(0x12, p);
		return;
	}
	p = tok_see();
	if (p[0] == '|' && p[1] == '=' && !p[2]) {
		tok_get();
		opassign(0x13, p);
		return;
	}
	p = tok_see();
	if (p[0] == '^' && p[1] == '=' && !p[2]) {
		tok_get();
		opassign(0x14, p);
		return;
	}
}
