#include "mbox.h"
#include "fn_xwrite.h"
#include <fcntl.h>
#include <stdlib.h>
#include <unistd.h>

int mbox_copy(struct mbox *mp, char *name)
{
	int fd, i;
	char *p;
	long n;

	fd = open(name, O_WRONLY | O_CREAT | O_TRUNC, 0600);
	if (fd < 0)
		return -1;
	i = mp->start[0];
	if (mp->end[0] > i)
		do {
			if (mp->c258[i]) {
				p = mp->c258[i];
				n = mp->c260[i];
			} else {
				p = mp->c248[i];
				n = mp->c250[i];
			}
			xwrite(fd, p, n);
			i++;
		} while (mp->end[0] > i);
	close(fd);
	return 0;
}
