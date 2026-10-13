#include "box.h"
#include "fn_nreg.h"
#include "fn_box_putf.h"
#include "fn_box_toreg.h"
#include "e_vars.h"

void box_matrix(struct box *b, int ncol, struct box **cells, int *align, int hs, int vs)
{
	int ra[32], rb[32];
	int w[32][32][4];
	int rb_t, ra_t;
	int n, i, j, k, m;

	rb_t = nregmk();
	ra_t = nregmk();
	box_beforeput(b, 0x80, 0);
	n = 0;
	if (ncol > 0) {
		for (i = 0; i < ncol; i++) {
			for (j = 0; j < 32 && cells[i * 32 + j]; j++)
				;
			if (j > n)
				n = j;
		}
		for (i = 0; i < ncol; i++)
			ra[i] = nregmk();
		for (i = 0; i < ncol; i++)
			rb[i] = nregmk();
		for (i = 0; i < ncol; i++)
			box_colinit(cells + 32 * i, n, w[i], ra[i], rb[i]);
	}
	printf(".nr %s 0%s\n", nregname(ra_t), nreg(ra[0]));
	printf(".nr %s 0%s\n", nregname(rb_t), nreg(rb[0]));
	if (ncol != 1 && ncol > 0) {
		for (i = 1; i < ncol; i++)
			printf(".if %s>%s .nr %s 0+%s\n", nreg(ra[i]), nreg(ra_t), nregname(ra_t), nreg(ra[i]));
		for (i = 1; i < ncol; i++)
			printf(".if %s>%s .nr %s 0+%s\n", nreg(rb[i]), nreg(rb_t), nregname(rb_t), nreg(rb[i]));
	}
	printf(".if %s<(%sp*%du/100u) .nr %s (%sp*%du/100u)\n", nregname(rb_t), nreg(b->type), e_baselinesep,
	       nregname(rb_t), nreg(b->type), e_baselinesep);
	if (vs)
		printf(".nr %s +(%sp*%du/100u)\n", nregname(rb_t), nreg(b->type), vs);
	for (k = 0; k < ncol; k++) {
		if (k)
			box_putf(b, "\\h'%sp*%du/100u'", nreg(b->type), hs + e_columnsep);
		box_colput(cells + 32 * k, n, b, align[k], w[k], ra_t, rb_t);
	}
	box_afterput(b, 0x80);
	box_toreg(b);
	for (k = 0; k < ncol; k++)
		for (i = 0; i < n; i++)
			for (m = 0; m < 4; m++)
				nregrm(w[k][i][m]);
	for (k = 0; k < ncol; k++)
		nregrm(rb[k]);
	for (k = 0; k < ncol; k++)
		nregrm(ra[k]);
	nregrm(ra_t);
	nregrm(rb_t);
}
