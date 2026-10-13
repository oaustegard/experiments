#include "env.h"

int tab_next(int x)
{
	int i;

	for (i = 0; i < 32; i++)
		if (x - env->tab[i] < 0)
			break;
	if (i < 32)
		return env->tab[i];
	return x;
}
