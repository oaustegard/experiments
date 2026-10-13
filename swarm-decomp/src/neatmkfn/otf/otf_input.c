#include <unistd.h>

static unsigned char *otf_input(int fd)
{
	struct sbuf *sb = sbuf_make();
	char buf[4096];
	int n;

	while ((n = read(fd, buf, sizeof buf)) > 0)
		sbuf_mem(sb, buf, n);
	return sbuf_done(sb);
}
