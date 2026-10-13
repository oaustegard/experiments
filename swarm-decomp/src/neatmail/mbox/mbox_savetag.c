#include "mbox.h"
#include "sys_fs.h"
#include "fn_xread.h"
#include "fn_xwrite.h"
#include "fn_mbox_savetag.h"

int mbox_savetag(struct mbox *m, int i)
{
	int j, k, fd, end;
	long total = 0, extra;
	struct stat st;
	char *buf = NULL;
	int size;

	if (!mbox_find(m, i))
		return 0;
	for (j = m->start[i]; j < m->end[i]; j++) {
		if (m->c258[j])
			break;
		total += m->c250[j];
	}
	fd = open(m->name[i], O_RDWR | O_CREAT, 0600);
	fstat(fd, &st);
	size = st.st_size;
	extra = size - m->off[i];
	if (extra > 0) {
		buf = malloc(extra);
		lseek(fd, m->off[i], SEEK_SET);
		xread(fd, buf, extra);
	}
	ftruncate(fd, lseek(fd, total, SEEK_SET));
	for (k = j; k < m->end[i]; k++) {
		char *p;
		long n;
		if (m->c258[k]) {
			p = m->c258[k];
			n = m->c260[k];
		} else {
			p = m->c248[k];
			n = m->c250[k];
		}
		lseek(fd, 0, SEEK_END);
		xwrite(fd, p, n);
	}
	if (extra > 0) {
		lseek(fd, 0, SEEK_END);
		xwrite(fd, buf, extra);
		free(buf);
	}
	close(fd);
	return extra;
}
