#include "libc.h"
#include "fn_dev_open.h"
#include "fn_dev_close.h"
#include "fn_hyph_init.h"
#include "fn_hyph_done.h"
#include "fn_env_init.h"
#include "fn_env_done.h"
#include "fn_tr_init.h"
#include "fn_tr_done.h"
#include "fn_map_done.h"
#include "fn_dir_done.h"
#include "fn_render.h"
#include "fn_in_queue.h"
#include "fn_nreg.h"
#include "fn_eval_re.h"
#include "fn_num.h"
#include "fn_str_set.h"
#include "fn_out.h"
#include "cmddef.h"
#include "cmdmac.h"

int main(int argc, char **argv)
{
	char *fdir, *mdir, *dev, *s, *name, *val;
	int i, r, ret;

	fdir = getenv("NEATROFF_F");
	mdir = getenv("NEATROFF_M");
	dev = getenv("NEATROFF_T");
	if (!dev)
		dev = "utf";
	if (!fdir)
		fdir = "/neatroff/font";
	if (!mdir)
		mdir = "/neatroff/tmac";

	for (i = 1; i < argc; i++) {
		s = argv[i];
		if (s[0] != '-' || s[1] == 0)
			break;
		switch (s[1]) {
		case 'C':
			*nreg('C') = 1;
			break;
		case 'F':
			if (s[2])
				fdir = s + 2;
			else
				fdir = argv[++i];
			break;
		case 'M':
			if (s[2])
				mdir = s + 2;
			else
				mdir = argv[++i];
			break;
		case 'T':
			if (s[2])
				dev = s + 2;
			else
				dev = argv[++i];
			break;
		case 'd':
			if (s[2])
				s = s + 2;
			else
				s = argv[++i];
			cmddef(s, &r, &val);
			str_set(r, val);
			break;
		case 'r':
			if (s[2])
				s = s + 2;
			else
				s = argv[++i];
			cmddef(s, &r, &val);
			num_set(r, eval_re(val, *nreg(r), 'u'));
			break;
		case 'm':
			name = s + 2;
			if (strchr(name, '/') || (cmdmac(mdir, name) && cmdmac(".", name)))
				in_queue(name);
			break;
		default:
			fwrite("Usage: neatroff [options] input\n\nOptions:\n  -mx   \tinclude macro x\n  -rx=y \tset number register x to y\n  -dx=y \tdefine string register x as y\n  -C    \tenable compatibility mode\n  -Tdev \tset output device\n  -Fdir \tset font directory (/neatroff/font)\n  -Mdir \tset macro directory (/neatroff/tmac)\n", 1, 0x127, stderr);
			return 1;
		}
	}
	if (dev_open(fdir, dev)) {
		fprintf(stderr, "neatroff: cannot open device %s\n", dev);
		return 1;
	}
	hyph_init();
	env_init();
	tr_init();
	if (i == argc) {
		in_queue(NULL);
	} else {
		for (; i < argc; i++) {
			s = argv[i];
			if (strcmp("-", s) == 0)
				s = NULL;
			in_queue(s);
		}
	}
	out("s%d\n", *nreg('s'));
	out("f%d\n", *nreg('f'));
	ret = render();
	out("V%d\n", *nreg('p'));
	hyph_done();
	tr_done();
	env_done();
	dev_close();
	map_done();
	dir_done();
	return ret;
}
