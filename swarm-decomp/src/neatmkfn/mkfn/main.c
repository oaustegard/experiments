#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int
main(int argc, char *argv[])
{
	int afm = 1;
	int i = 1;
	char *s;

	for (; i < argc; i++) {
		s = argv[i];
		if (s[0] != '-')
			break;
		switch (s[1]) {
		case 'a':
			afm = 1;
			break;
		case 'o':
			afm = 0;
			break;
		case 'b':
			mkfn_bbox = 1;
			break;
		case 'f':
			mkfn_path = s[2] ? s + 2 : argv[++i];
			break;
		case 'g':
			mkfn_byname = 1;
			break;
		case 'k':
			mkfn_kmin = strtol(s[2] ? s + 2 : argv[++i], NULL, 10);
			break;
		case 'l':
			mkfn_noligs = 1;
			break;
		case 'n':
			mkfn_pos = 0;
			break;
		case 'p':
			mkfn_psname = s[2] ? s + 2 : argv[++i];
			break;
		case 'r':
			mkfn_res = strtol(s[2] ? s + 2 : argv[++i], NULL, 10);
			break;
		case 's':
			mkfn_special = 1;
			break;
		case 't':
			mkfn_trname = s[2] ? s + 2 : argv[++i];
			break;
		case 'w':
			mkfn_warn = 1;
			break;
		case 'F':
			s = s[2] ? s + 2 : argv[++i];
			mkfn_subfont = s;
			mkfn_dry = strcmp("list", s) == 0;
			break;
		case 'L':
			s = s[2] ? s + 2 : argv[++i];
			mkfn_langs = s;
			mkfn_dry = strcmp("list", s) == 0;
			break;
		case 'S':
			s = s[2] ? s + 2 : argv[++i];
			mkfn_scripts = s;
			mkfn_dry = strcmp("list", s) == 0;
			break;
		default:
			fwrite("Usage: mkfn [options] <input >output\nOptions:\n  -a      \tread an AFM file (default)\n  -o      \tread a TTF or an OTF file\n  -s      \tspecial font\n  -p name \toverride font postscript name\n  -t name \tset font troff name\n  -f path \tset font path\n  -r res  \tset device resolution (720)\n  -k kmin \tspecify the minimum amount of kerning (0)\n  -b      \tinclude glyph bounding boxes\n  -l      \tsuppress the ligatures line\n  -n      \tsuppress glyph positions\n  -g      \talways reference glyphs by name in OTF rules\n  -S scrs \tcomma-separated list of scripts to include (list to list)\n  -L langs\tcomma-separated list of languages to include (list to list)\n  -F font \tfont name or index in a font collection (list to list)\n  -w      \twarn about unsupported font features\n",
			1, 0x2f7, stderr);
			return 1;
		}
	}
	trfn_init();
	if (afm) {
		if (afm_read())
			goto error;
	} else if (otf_read())
		goto error;
	trfn_done();
	return 0;
error:
	fwrite("neatmkfn: cannot parse the font\n", 1, 0x20, stderr);
	trfn_done();
	return 1;
}
