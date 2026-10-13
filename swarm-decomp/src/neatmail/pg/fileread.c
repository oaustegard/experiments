#include <errno.h>
#include <fcntl.h>
#include <unistd.h>
#include "fn_sbuf.h"

static char *fileread(char *path, int *len)
{
	char buf[0x1000];
	struct sbuf *sb;
	int fd, n;

	fd = open(path, 0);
	if (fd < 0)
		return NULL;
	sb = sbuf_make();
	for (;;) {
		n = read(fd, buf, sizeof(buf));
		if (n == -1 && (errno == EAGAIN || errno == EINTR))
			continue;
		if (n <= 0)
			break;
		sbuf_mem(sb, buf, n);
	}
	close(fd);
	*len = sbuf_len(sb);
	return sbuf_done(sb);
}
