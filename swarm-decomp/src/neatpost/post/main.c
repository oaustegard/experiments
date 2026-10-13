#include <ctype.h>

int main(int argc, char **argv)
{
	int i, c, land = 0, k, n, k1, k2;
	int A, B, first, second, lc;
	char *s, *p;

	if (getenv("NEATROFF_F") != 0)
		snprintf(postdir, 0x400, "%s", getenv("NEATROFF_F"));
	for (i = 1; i < argc; i++) {
		s = argv[i];
		if (s[0] != '-')
			break;
		switch (s[1]) {
		case 'F':
			s = s[2] ? s + 2 : argv[++i];
			strcpy(postdir, s);
			break;
		case 'd':
			s = s[2] ? s + 2 : argv[++i];
			p = strchr(s, '=');
			if (p) {
				*p = 0;
				outset(s, p + 1);
			}
			break;
		case 'l':
			land = 1;
			break;
		case 'n':
			outgname(1);
			break;
		case 'p':
			s = s[2] ? s + 2 : argv[++i];
			for (k = 0; k < 4; k++)
				if (strcmp(papers[k].name, s) == 0)
					break;
			if (k < 4) {
				ps_pagewidth = papers[k].w;
				ps_pageheight = papers[k].h;
			} else if (isdigit(s[0]) && strchr(s, 'x') != 0) {
				ps_pagewidth = strtol(s, 0, 10);
				ps_pageheight = strtol(strchr(s, 'x') + 1, 0, 10);
			} else if (strchr("abcABC", s[0]) != 0 && isdigit(s[1])) {
				n = s[1];
				lc = tolower(s[0]);
				if (lc == 'b') {
					A = 0x373c;
					B = 0x2710;
				} else if (lc == 'c') {
					A = 0x32aa;
					B = 0x23d2;
				} else {
					A = 0x2e72;
					B = 0x20da;
				}
				k1 = (n - 0x2f) >> 1;
				k2 = (n - 0x30) >> 1;
				if (n & 1) {
					first = A >> k1;
					second = B >> k2;
				} else {
					first = B >> k1;
					second = A >> k2;
				}
				ps_pagewidth = first - first % 10;
				ps_pageheight = second - second % 10;
			}
			break;
		case 't':
			ps_title = s[2] ? s + 2 : argv[++i];
			break;
		case 'w':
			ps_linewidth = strtol(s[2] ? s + 2 : argv[++i], 0, 10);
			break;
		default:
			fwrite("Usage: neatpost [options] <input >output\nOptions:\n  -F dir  \tset font directory (/neatroff/font)\n  -p size \tset paper size (letter); e.g., a4, 2100x2970\n  -t title\tspecify document title\n  -w lwid \tdrawing line thickness in thousandths of an em (40)\n  -l      \tlandscape mode\n  -n      \talways draw glyphs by name (ps glyphshow)\n  -d x=v  \tset device-specific variables\n", 1, 0x172, stderr);
			return 1;
		}
	}
	if (land) {
		k = ps_pagewidth;
		ps_pagewidth = ps_pageheight;
		ps_pageheight = k;
	}
	while ((c = getc(stdin)) >= 0) {
		if (!isspace(c))
			postcmd(c);
	}
	if (o_pages)
		docpageend();
	if (name_n)
		outname(name_n, name_desc, name_page, name_offset);
	if (mark_n)
		outmark(mark_n, mark_desc, mark_page, mark_offset, mark_level);
	doctrailer(o_pages);
	dev_close();
	free(mark_desc);
	free(mark_page);
	free(mark_offset);
	free(mark_level);
	free(name_desc);
	free(name_page);
	free(name_offset);
	return 0;
}
