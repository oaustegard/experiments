#include "ren_line.h"
#include "fmt_morewords.h"
#include "ren_page.h"

static int ren_passline(void *fmt)
{
	int r12, p4, p2, p3, p5, p6, p1, r;
	char *p;

	r12 = *nreg(0x6a);
	if (bp_first && !cdiv) {
		bp_first = 0;
		ren_page(1);
	}
	if (!fmt_morewords(fmt))
		return 0;
	p = fmt_nextline(fmt, &p1, &p2, &p3, &p4, &p5, &p6);
	if (*nreg(0x75)) {
		r12 &= 7;
		if (r12)
			goto ce;
		goto td;
	}
	if (*nreg(map(".na")))
		goto td;
	if (*nreg(0x43) == 0) {
		if ((r12 & 0x700) == 0)
			goto td;
		if ((r12 & 0x700) == 0x200) {
			r12 = 4;
			goto ce;
		}
		if ((r12 & 0x700) == 0x400) {
			r12 = 1;
			goto ce;
		}
		goto ce;
	}
td:
	if (*nreg(map(".td")) > 0)
		r12 = 4;
	else
		r12 = 2;
ce:
	if (*nreg(map(".ce")))
		r12 = 1;
	r = ren_line(p, p1, r12, 1, p2, p3, p4, p5, p6);
	free(p);
	return r;
}
