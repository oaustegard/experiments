#include "fn_cp_next.h"
#include "fn_tr_nextreq.h"

int tr_next(void)
{
	int c;

	while (!tr_nextreq())
		;
	c = cp_next();
	tr_nl = c == 10 || c < 0;
}
