#include <ctype.h>

static void postcmd(int c)
{
	char w[128], u[128];
	int ch, n, cmd;

	if (isdigit(c)) {
		n = (c - '0') * 10 + (getc(stdin) - '0');
		outrel(n, 0);
		nextutf8(u, c);
		outc(u);
		return;
	}
	switch (c) {
	case '#':
		while ((ch = getc(stdin)) >= 0 && ch != '\n')
			;
		return;
	case 'C':
		nextword(u);
		outc(u);
		return;
	case 'D': {
		int a, b;

		ch = getc(stdin);
		drawbeg();
		switch (tolower(ch)) {
		case 'a':
			postarc();
			break;
		case 'c':
			drawc(nextnum());
			break;
		case 'e':
			a = nextnum();
			b = nextnum();
			drawe(a, b);
			break;
		case 'l':
			a = nextnum();
			b = nextnum();
			drawl(a, b);
			break;
		case '~':
			postspline();
			break;
		case 'p':
			cmd = 'l';
			for (;;) {
				do
					c = getc(stdin);
				while (c == ' ');
				ungetc(c, stdin);
				if (c == '\n' || (unsigned)(cmd - 'a') > 29 || !((0x20000801ul >> (cmd - 'a')) & 1))
					break;
				do
					c = getc(stdin);
				while (c == ' ');
				ungetc(c, stdin);
				if (c == '+' || c == '-' || isdigit(c)) {
					if (cmd == 'l') {
						for (;;) {
							if (readnum(&a))
								break;
							if (readnum(&b))
								break;
							drawl(a, b);
						}
					} else if (cmd == '~') {
						postspline();
					} else {
						postarc();
						cmd = 'a';
					}
				} else {
					n = c;
					while (n >= 0 && !isspace(n))
						n = getc(stdin);
					cmd = c;
				}
			}
			break;
		}
		drawend((ch & ~0x20) == 'P', ch == 'C' || ch == 'E' || ch == 'P');
		goto eol;
	}
	case 'H':
		outh(nextnum());
		return;
	case 'N':
		nextnum();
		return;
	case 'V':
		outv(nextnum());
		return;
	case 'c':
		nextutf8(u, c);
		outc(u);
		return;
	case 'f':
		outfont(nextnum());
		return;
	case 'h': {
		int a;

		a = nextnum();
		outrel(a, 0);
		return;
	}
	case 'm':
		nextword(u);
		outcolor(clr_get(u));
		return;
	case 'n':
		nextnum();
		nextnum();
		return;
	case 'p':
		if (o_pages)
			docpageend();
		o_pages = nextnum();
		docpagebeg(o_pages);
		outpage();
		return;
	case 's':
		outsize(nextnum());
		return;
	case 'v': {
		int a;

		a = nextnum();
		outrel(0, a);
		return;
	}
	case 'w':
		return;
	case 'x':
		nextword(w);
		switch (w[0]) {
		case 'f':
			n = nextnum();
			nextword(u);
			dev_mnt(n, u, u);
			outmnt(n);
			break;
		case 'i':
			if (dev_open(postdir, postdev) == 0)
				docheader(ps_title, ps_pagewidth, ps_pageheight, ps_linewidth);
			else {
				fprintf(stderr, "neatpost: cannot open device %s\n", postdev);
				exit(1);
			}
			break;
		case 'T':
			nextword(postdev);
			break;
		case 'X':
			postps();
			break;
		}
		goto eol;
	default:
		fprintf(stderr, "neatpost: unknown command %c\n", c);
		while ((ch = getc(stdin)) >= 0 && ch != '\n')
			;
		return;
	}
eol:
	for (;;) {
		ch = getc(stdin);
		if (ch < 0 || ch == '\n')
			return;
	}
}
