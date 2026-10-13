#include "fmt.h"
#include "wb.h"
#include "fn_map.h"
#include "fn_nreg.h"
#include "fn_wb_empty.h"
#include "fn_wb_eos.h"

static int fmt_confchanged(struct fmt *f);
static int fmt_fillwords(struct fmt *f, int n);
static int fmt_flush(struct fmt *f);
static void fmt_insertword(struct fmt *f, struct wb *w, int sp);
static void fmt_confupdate(struct fmt *f);
static int fmt_wordgap(struct fmt *f);

int fmt_word(struct fmt *f, struct wb *w)
{
	int e = wb_empty(w);

	if (e)
		return 0;
	if (fmt_confchanged(f)) {
		if (fmt_fillwords(f, 0))
			return 1;
	}
	if (*nreg(map(".ce")) == 0) {
		if (*nreg('u') && f->nosuppress && f->spacing && fmt_flush(f))
			return 1;
	}
	if (f->nwords == 0)
		fmt_confupdate(f);
	f->spacing = fmt_wordgap(f);
	f->f5c = wb_eos(w);
	if (f->f58 == 0)
		e = f->spacing;
	fmt_insertword(f, w, e);
	f->f58 = 0;
	f->f48 = 0;
	f->spacing = 0;
	f->nosuppress = 0;
	return 0;
}
