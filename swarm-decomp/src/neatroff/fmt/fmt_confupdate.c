#include "fmt.h"
#include "fn_map.h"
#include "fn_nreg.h"

static void fmt_confupdate(struct fmt *f)
{
	f->f50 = *nreg('l');
	if (*nreg(map(".ti")) < 0)
		f->f4c = *nreg('i');
	else
		f->f4c = *nreg(map(".ti"));
	if (*nreg(map(".tI")) < 0)
		f->f54 = *nreg('I');
	else
		f->f54 = *nreg(map(".tI"));
	*nreg(map(".ti")) = -1;
	*nreg(map(".tI")) = -1;
}
