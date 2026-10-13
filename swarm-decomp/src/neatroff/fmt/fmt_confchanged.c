#include "fmt.h"
#include "fn_map.h"
#include "fn_nreg.h"

/* 0 when the .ti and .tI registers still hold the values recorded in f;
   1 when the configuration changed (fmt_confupdate records it again) */
static int fmt_confchanged(struct fmt *f)
{
	struct { int v; } a, b;

	if (f->f50 == *nreg('l')) {
		a.v = f->f4c;
		if (*nreg(map(".ti")) < 0)
			b.v = *nreg('i');
		else
			b.v = *nreg(map(".ti"));
		if (a.v != b.v)
			return 1;
		a.v = f->f54;
		if (*nreg(map(".tI")) < 0)
			b.v = *nreg('I');
		else
			b.v = *nreg(map(".tI"));
		return a.v != b.v;
	}
	return 1;
}
