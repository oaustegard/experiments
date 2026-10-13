#include "fmt.h"
#include "fn_map.h"
#include "fn_nreg.h"
#include "fn_dev_font.h"
#include "fn_font_swid.h"
#include "fmt_wordgap.h"

static int fmt_wordgap(struct fmt *f)
{
	int swid, sp, ss, s, fn, sss, f48;

	if (f->nosuppress)
		goto nosup;
	f48 = f->f48;
	ss = *nreg(map(".ss"));
	s = *nreg('s');
	fn = *nreg('f');
	swid = font_swid(dev_font(fn), s, ss);
	if (!f->f5c || !f->nwords)
		goto bg;
	sp = f->spacing;
	if (f48 != 0)
		goto bi;
	if (2 * swid != sp)
		return sp;
	goto sss;
bg:
	sp = f->spacing;
	if (f48 != 0)
		goto bh;
	return sp;
bi:
	if (sp)
		return sp;
	goto sss;
nosup:
	ss = *nreg(map(".ss"));
	s = *nreg('s');
	fn = *nreg('f');
	swid = font_swid(dev_font(fn), s, ss);
	sp = f->spacing;
	if (!f->f5c || !f->nwords)
		goto bh;
	goto bi;
bh:
	if (sp)
		return sp;
	return f->nwords ? swid : 0;
sss:
	sss = *nreg(map(".sss"));
	s = *nreg('s');
	fn = *nreg('f');
	return font_swid(dev_font(fn), s, sss) + swid;
}
