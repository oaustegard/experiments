#include "fmt.h"
#include "fn_map.h"
#include "fn_nreg.h"
#include "fmt_flush.h"
#include "fmt_extractline.h"

int fmt_newline(struct fmt *f)
{
	f->spacing = 0;
	if (*nreg(map(".ce")) != 0 || *nreg('u') == 0) {
		f->nosuppress++;
		fmt_flush(f);
		return 0;
	}
	if (f->nosuppress) {
		if (fmt_flush(f))
			return 1;
	}
	if (f->nosuppress == 0 && (f->f58 != 0 || f->nwords != 0))
		goto skip;
	if (fmt_extractline(f, f->nwords, 0))
		return 1;
skip:
	f->nosuppress++;
	return 0;
}
