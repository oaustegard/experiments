#include <string.h>
#include "env.h"
#include "tr_args.h"
#include "fn_eval_re.h"

void tr_ta(struct tr_args *a)
{
	int i;
	int c;
	char *s;

	for (i = 0; i < 31; i++) {
		s = a->arg[i];
		if (s) {
			c = 0;
			env->tab[i] = eval_re(s, i ? env->tab[i - 1] : 0, 'm');
			if (*s)
				c = (unsigned char)s[strlen(s) - 1];
			if (!strchr("LRC", c))
				c = 'L';
			env->pad180[i] = c;
		} else {
			env->tab[i] = 0;
			env->pad180[i] = 0;
		}
	}
	env->tab[31] = 0;
	env->pad180[31] = 0;
}
