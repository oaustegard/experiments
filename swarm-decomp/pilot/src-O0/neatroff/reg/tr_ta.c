#include <string.h>

void tr_ta(char **arg)
{
	int i, k;
	unsigned char *p;
	for (i = 0; i < 32; i++) {
		if (i <= 30 && arg[i + 1]) {
			p = (unsigned char *)arg[i + 1];
			env->ints[(long)i + 0x40] = eval_re((char *)p, i > 0 ? env->ints[(long)(i - 1) + 0x40] : 0, 'm');
			k = *p ? p[strlen((char *)p) - 1] : 0;
			env->chars[i] = strchr("LRC", k) ? k : 'L';
		} else {
			env->ints[(long)i + 0x40] = 0;
			env->chars[i] = 0;
		}
	}
}
