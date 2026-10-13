#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#include <utime.h>
#include "mbox.h"
#include "fn_xread.h"
#include "fn_mbox_free.h"

struct mbox *mbox_open(char **argv)
{
	struct mbox *m;
	char **ap;
	char *s, *buf, *end, *p, *q, *np;
	int i, fd, l, nc, inc;
	int size;
	struct stat st;
	struct utimbuf ut;

	m = calloc(sizeof *m, 1);
	if (!m)
		return 0;
	for (ap = argv, s = *ap; s; s = *++ap) {
		i = m->n;
		if (i > 14)
			continue;
		m->n = i + 1;
		l = strlen(s) + 1;
		m->name[i] = malloc(l);
		if (!m->name[i])
			goto fail;
		memcpy(m->name[i], s, l);
		fd = open(s, O_RDONLY);
		if (fd < 0)
			goto fail;
		fstat(fd, &st);
		size = st.st_size;
		m->off[i] = size;
		m->name[i + 16] = malloc(size + 1);
		buf = m->name[i + 16];
		if (!buf)
			goto fail;
		xread(fd, buf, size);
		buf[m->off[i]] = 0;
		close(fd);
		stat(m->name[i], &st);
		ut.actime = time(0);
		ut.modtime = st.st_mtime;
		utime(m->name[i], &ut);
		nc = m->nc;
		m->start[i] = nc;
		buf = m->name[i + 16];
		end = buf + m->off[i];
		p = mbox_from_(buf, end);
		if (p && p < end) {
			for (;;) {
				if (m->pad268 == nc) {
					inc = nc ? nc : 256;
					m->pad268 = nc + inc;
					np = malloc((long)(nc + inc) * 8);
					memcpy(np, m->c248, nc * 8);
					memset(np + nc * 8, 0, inc * 8);
					free(m->c248);
					m->c248 = (char **)np;
					np = malloc((long)m->pad268 * 8);
					memcpy(np, m->c250, nc * 8);
					memset(np + nc * 8, 0, (m->pad268 - nc) * 8);
					free(m->c250);
					m->c250 = (long *)np;
					if (!m->c248)
						goto fail;
				}
				m->c248[nc] = p;
				nc++;
				q = mbox_from_(p + 6, end);
				if (!q) {
					m->c250[nc - 1] = end - p;
					m->nc = nc;
					break;
				}
				m->c250[nc - 1] = q - p;
				m->nc = nc;
				if (q >= end)
					break;
				p = q;
			}
		}
		m->end[i] = nc;
	}
	m->c258 = calloc(m->nc, 8);
	m->c260 = calloc(m->nc, 8);
	if (!m->c258 || !m->c260)
		goto fail;
	return m;
fail:
	mbox_free(m);
	return 0;
}
