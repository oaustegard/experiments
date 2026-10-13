#include <stdio.h>
#include <string.h>
#include "outf.h"
#include "ps_state.h"
#include "ps_extern.h"
#include "ps_vars.h"

void outeps(char *name, int w, int h)
{
	FILE *f;
	char buf[4096];
	int b0, b1, b2, b3, dummy, n;

	f = fopen(name, "r");
	if (!f)
		return;
	if (!fgets(buf, sizeof buf, f)) {
		fprintf(stderr, "warning: file %s is empty\n", name);
		fclose(f);
		return;
	}
	if (sscanf(buf, "%%!PS-Adobe-%d.%d EPSF-%d.%d", &dummy, &dummy, &dummy, &dummy) != 4) {
		fprintf(stderr, "warning: unsupported EPSF header in %s\n", name);
		fclose(f);
		return;
	}
	n = 0;
	while (fgets(buf, sizeof buf, f)) {
		if (strncmp(buf, "%%BoundingBox: ", 15) == 0) {
			n = sscanf(buf + 15, "%d %d %d %d", &b0, &b1, &b2, &b3);
			if (n == 4)
				break;
		}
	}
	fclose(f);
	if (n <= 3)
		return;
	if (w > 0) {
		if (h <= 0)
			h = (b3 - b1) * w / (b2 - b0);
	} else {
		if (h <= 0) {
			w = dev_res * (b2 - b0) / 72;
			h = (b3 - b1) * w / (b2 - b0);
		}
		if (w <= 0)
			w = (b2 - b0) * h / (b3 - b1);
	}
	o_flush();
	out_fontup(o_f);
	outf("%d %d %d %d %d %d %d %d EPSFBEG\n", b0, b1, w, b2 - b0, h, b3 - b1, o_h, o_v);
	outf("%%%%BeginDocument: %s\n", name);
	f = fopen(name, "r");
	while (fgets(buf, sizeof buf, f))
		out("%s", buf);
	fclose(f);
	outf("%%%%EndDocument\n");
	outf("EPSFEND\n");
}
