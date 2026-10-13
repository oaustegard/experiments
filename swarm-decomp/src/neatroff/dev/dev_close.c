#include "fn_out.h"
#include "fn_font_close.h"

void dev_close(void)
{
	int i;

	out("x trailer\n");
	out("x stop\n");
	for (i = 0; i < 32; i++) {
		if (fn_font[i])
			font_close(fn_font[i]);
		fn_font[i] = NULL;
	}
}
