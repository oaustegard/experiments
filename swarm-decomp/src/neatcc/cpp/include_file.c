#include <fcntl.h>
#include <sys/stat.h>
#include "cpp_protos.h"
#include "cpp_state.h"

static int include_file(char *name)
{
	struct stat st;
	int fd;
	int size;
	int off;
	int n;
	char *p;

	fd = open(name, 0);
	if (fd == -1)
		return fd;
	size = 1;
	if (fstat(fd, &st) != 0)
		;
	else
		size = st.st_size + 1;
	p = malloc(size);
	off = 0;
	while ((n = read(fd, p + off, size - off)) > 0)
		off += n;
	close(fd);
	fd = 0;
	p[off] = 0;
	buf_new(0, p, off);
	strcpy(bufs[bufs_n - 1].rest, name);
	return fd;
}
