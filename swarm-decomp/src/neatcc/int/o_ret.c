#include "ic_put.h"

/* return: with n == 0 an empty value is pushed first; pop the value and emit a return */
void o_ret(int n)
{
	if (!n)
		ic_put(0x100400, 0, 0, 0);
	ic_put(0x8000, iv[--iv_n], 0, 0);
}
