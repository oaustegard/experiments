#include <fcntl.h>
#include <poll.h>
#include <stdlib.h>
#include <sys/wait.h>
#include <unistd.h>
#include "fn_sbuf.h"
#include "fn_xpipe.h"

int
xpipe(char *cmd, char *in, long n, char **out, long *outn)
{
	char *argv[4] = {"/bin/sh", "-c", cmd, NULL};
	struct sbuf *sb;
	int rfd, wfd, pid, off;

	{
		int a[2], b[2];

		if (out) {
			if (in)
				pipe(a);
			pipe(b);
		} else if (in)
			pipe(a);
		pid = fork();
		if (pid == 0) {
			if (in) {
				close(0);
				dup(a[0]);
				close(a[1]);
				close(a[0]);
			}
			if (out) {
				close(1);
				dup(b[1]);
				close(b[0]);
				close(b[1]);
			}
			execvp(argv[0], argv);
			exit(1);
		}
		if (in)
			close(a[0]);
		if (out)
			close(b[1]);
		if (pid < 0) {
			if (in)
				close(a[1]);
			if (out)
				close(b[0]);
			return 1;
		}
		wfd = in ? a[1] : -1;
		rfd = out ? b[0] : -1;
	}
	sb = out ? sbuf_make() : NULL;
	fcntl(wfd, F_SETFL, fcntl(wfd, F_GETFL, 0) | O_NONBLOCK, 0);
	{
		struct pollfd fds[2];
		char buf[512];
		int r, w;

		fds[0].fd = rfd;
		fds[0].events = POLLIN;
		fds[1].fd = wfd;
		fds[1].events = POLLOUT;
		off = 0;
		while ((fds[0].fd & fds[1].fd) >= 0) {
			if (poll(fds, 2, 200) < 0)
				break;
			if (fds[0].revents & POLLIN) {
				r = read(fds[0].fd, buf, sizeof buf);
				if (r <= 0) {
					close(fds[0].fd);
					fds[0].fd = -1;
				} else
					sbuf_mem(sb, buf, r);
			} else if (fds[0].revents & 0x38)
				fds[0].fd = -1;
			if (fds[1].revents & POLLOUT) {
				w = write(fds[1].fd, in + off, n - off);
				if (w > 0)
					off += w;
				if (w <= 0 || off == n) {
					close(fds[1].fd);
					fds[1].fd = -1;
				}
			} else if (fds[1].revents & 0x38)
				fds[1].fd = -1;
		}
	}
	close(wfd);
	close(rfd);
	waitpid(pid, NULL, 0);
	if (out) {
		*outn = sbuf_len(sb);
		*out = sbuf_done(sb);
	}
	return 0;
}
