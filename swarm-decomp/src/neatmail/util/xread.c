#include <errno.h>
#include <unistd.h>

long
xread(int fd, char *buf, long n)
{
	int r;
	long done = 0;
	int tot = 0;
	long pos = 0;

	while (pos - n < 0) {
		pos = tot;
		r = read(fd, buf + pos, n - done);
		if (r == -1 && (errno == EAGAIN || errno == EINTR))
			;
		else if (r <= 0)
			break;
		else
			tot += r;
		pos = tot;
		done = pos;
	}
	return done;
}
