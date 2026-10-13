#include "font.h"
#include "dev_uwid.h"

int font_swid(struct font *f, int n)
{
	return (f->scale * n + dev_uwid / 2) / dev_uwid;
}
