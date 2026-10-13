#include "outf.h"
#include "ps_extern.h"
#include "ps_vars.h"

void outmark(int n, char (*title)[256], int *page, int *y, int *lev)
{
	int i, j, cnt;

	o_flush();
	for (i = 0; i < n; i++) {
		cnt = 0;
		for (j = i + 1; j < n && lev[j] > lev[i]; j++)
			cnt += lev[j] == lev[i] + 1;
		outf("[ /Title %s", pdftext_static(title[i]));
		if (page[i] > 0)
			outf(" /Page %d", page[i]);
		if (cnt > 0)
			outf(" /Count %d", cnt);
		if (y[i] > 0)
			outf(" /View [/XYZ null %d null]", (ps_height - y[i]) * 72 / dev_res);
		outf(" /OUT pdfmark\n");
	}
}
