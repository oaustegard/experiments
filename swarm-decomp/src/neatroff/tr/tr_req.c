#include "tr_args.h"
#include "fn_str_dget.h"

void tr_req(int r, struct tr_args *a)
{
	char *p = str_dget(r);

	if (p)
		(*(void (**)(struct tr_args *))(p + 8))(a);
}
