#include "font.h"
#include "dev_uwid.h"

int font_wid(struct font *f, int n, int w)
{
	return (w * n + dev_uwid / 2) / dev_uwid;
}
