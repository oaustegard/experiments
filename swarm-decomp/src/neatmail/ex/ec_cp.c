#include <fcntl.h>
#include <stdlib.h>
#include <unistd.h>
#include "fn_mbox_get.h"
#include "fn_xwrite.h"
#include "ex_arg.h"
#include "ec_cp.h"

static int ec_cp(char *s)
{
	char name[512];
	char *p;
	long n;
	int fd, r;

	ex_arg(s, name);
	r = mbox_get(mbox, pos, &p, &n);
	if (r)
		return r;
	fd = open(name, O_WRONLY | O_CREAT | O_APPEND, 0600);
	if (fd < 0)
		return fd;
	xwrite(fd, p, n);
	return close(fd);
}
