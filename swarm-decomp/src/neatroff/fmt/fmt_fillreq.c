#include "fmt.h"
#include "fmt_fillwords.h"

int fmt_fillreq(struct fmt *f)
{
	if (f->fillreq > 0) {
		if (fmt_fillwords(f, 0))
			return 1;
	}
	f->fillreq = f->nwords + 1;
	return 0;
}
