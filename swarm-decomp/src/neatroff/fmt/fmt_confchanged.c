#include "fmt.h"
#include "fn_map.h"
#include "fn_nreg.h"

/* 0 when the .ti and .tI registers still hold the values recorded in f;
   1 when the configuration changed (fmt_confupdate records it again) */
static int fmt_confchanged(struct fmt *f)
{
	int a, b;

	if (f->f50 == *nreg('l')) {
		a = f->f4c;
		if (*nreg(map(".ti")) < 0)
			b = *nreg('i');
		else
			b = *nreg(map(".ti"));
		if (b == a) {
			a = f->f54;
			if (*nreg(map(".tI")) < 0)
				b = *nreg('I');
			else
				b = *nreg(map(".tI"));
			return b != a;
		}
	}
	return 1;
}
