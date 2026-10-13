#include "ps_extern.h"

void docpageend(void)
{
	out("cleartomark\n");
	out("showpage\n");
	out("saveobj restore\n");
}
