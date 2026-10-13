#include "mbox.h"
#include <stdlib.h>

void mbox_free(struct mbox *mp)
{
	int i;

	for (i = 0; i < mp->n; i++) {
		free(mp->name[i]);
		free(mp->name[i + 16]);
	}
	if (mp->c258)
		for (i = 0; i < mp->nc; i++)
			free(mp->c258[i]);
	free(mp->c248);
	free(mp->c250);
	free(mp->c258);
	free(mp->c260);
	free(mp);
}
