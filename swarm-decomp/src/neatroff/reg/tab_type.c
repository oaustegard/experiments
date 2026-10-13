#include "env.h"

int tab_type(int x)
{
	int i;
	char t;

	i = 0;
	while (x - env->tab[i] >= 0) {
		if (++i == 32)
			return 'L';
	}
	t = env->pad180[i];
	return t ? t : 'L';
}
