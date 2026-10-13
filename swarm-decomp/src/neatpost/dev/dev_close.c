#include "font.h"
#include "fn_font_close.h"

void dev_close(void)
{
	int i;
	for (i = 0; i < 1024; i++) {
		if (fn_font[i])
			font_close(fn_font[i]);
		fn_font[i] = 0;
	}
}
