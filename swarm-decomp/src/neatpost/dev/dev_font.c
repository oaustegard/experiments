#include "font.h"
#include "fn_dev_font.h"

void *dev_font(int i)
{
	void *r = 0;
	if ((unsigned int)i <= 1023)
		r = fn_font[i];
	return r;
}
