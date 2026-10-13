#include <errno.h>
#include <unistd.h>

long
xread(int fd, char *buf, long n)
{
	int r, tot = 0;
	long done = 0, pos;

	while (n > 0) {
		pos = tot;
		r = read(fd, buf + pos, n - done);
		if (r == -1) {
			if (errno != EAGAIN && errno != EINTR)
				break;
		} else if (r <= 0)
			break;
		else
			tot += r;
		pos = tot;
		done = pos;
		if (pos >= n)
			break;
	}
	return done;
}
