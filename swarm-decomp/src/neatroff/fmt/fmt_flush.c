#include "fmt.h"
#include "fn_map.h"
#include "fn_nreg.h"
#include "fmt_fillwords.h"

static int fmt_extractline(struct fmt *f, int n, int b);

static int fmt_flush(struct fmt *f)
{
	if (*nreg(map(".ce")) == 0 && *nreg('u') != 0 && fmt_fillwords(f, 1))
		return 1;
	if (*nreg(map(".ce")) != 0 || *nreg('u') == 0) {
		if (fmt_extractline(f, f->nwords, 0))
			return 1;
	}
	f->f58 = 0;
	f->f48 = 0;
	f->fillreq = 0;
	return 0;
}
