#include "font.h"

int dev_fontid(struct font *f)
{
	long i;
	for (i = 0; i < 1024; i++)
		if (fn_font[i] == f)
			return i;
	return 0;
}
