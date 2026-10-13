#include "wb.h"
#include "fn_wb_init.h"
#include "fn_wb_putexpand.h"
#include "fn_wb_wid.h"
#include "fn_wb_done.h"

static int cwid(char *s)
{
	struct wb w;
	int n;

	wb_init(&w);
	wb_putexpand(&w, s);
	n = wb_wid(&w);
	wb_done(&w);
	return n;
}
