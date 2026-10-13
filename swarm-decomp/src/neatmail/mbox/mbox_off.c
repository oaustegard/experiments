#include "mbox.h"
#include "fn_xread.h"
#include <fcntl.h>
#include <stdlib.h>
#include <unistd.h>

int mbox_off(char *name, long off, long len, char **s, long *n)
{
	int fd;
	long r;

	fd = open(name, O_RDONLY);
	if (fd < 0)
		return 1;
	if (lseek(fd, off, SEEK_SET) < 0) {
		close(fd);
		return 1;
	}
	len -= off;
	*s = malloc(len + 1);
	if (!*s)
		return 1;
	r = xread(fd, *s, len);
	(*s)[r] = 0;
	*n = r;
	close(fd);
	return 0;
}
