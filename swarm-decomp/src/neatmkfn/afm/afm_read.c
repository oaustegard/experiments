#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mkfn_ext.h"

int
afm_read(void)
{
	char fontname[128];
	char name[256] = {0};
	char code[256] = {0};
	char lig1[256] = {0}, lig2[256] = {0};
	char wx[256] = {0};
	char key[256] = {0};
	char bb[4][256] = {{"0"}, {"0"}, {"0"}, {"0"}};
	char line[1024];
	char *p;
	int k, n, c, cnt;

	for (;;) {
		if (fgets(line, 1024, stdin) == NULL)
			break;
		if (line[0] == '#')
			continue;
		if (strncmp(line, "FontName", 8) == 0) {
			sscanf(line, "FontName %s", fontname);
			continue;
		}
		if (strncmp(line, "StartCharMetrics", 16) == 0)
			break;
	}
	for (;;) {
		if (fgets(line, 1024, stdin) == NULL)
			break;
		if (line[0] == '#')
			continue;
		if (strncmp(line, "EndCharMetrics", 14) == 0)
			break;
		p = line;
		while (*p) {
			p = afm_charfield(p, key);
			if (key[0] == 'C' && key[1] == 0) {
				p = afm_charfield(p, code);
			} else if (key[0] == 'W' && key[1] == 'X' && key[2] == 0) {
				p = afm_charfield(p, wx);
			} else if (key[0] == 'N' && key[1] == 0) {
				p = afm_charfield(p, name);
			} else if (key[0] == 'B' && key[1] == 0) {
				for (n = 0; n < 4; n++)
					p = afm_charfield(p, bb[n]);
			} else if (key[0] == 'L' && key[1] == 0) {
				p = afm_charfield(p, lig1);
				p = afm_charfield(p, lig2);
			} else {
				break;
			}
		}
		if (name[0] == 0 || code[0] == 0 || wx[0] == 0)
			continue;
		k = 72000 / mkfn_res;
		c = (int)strtol(code, NULL, 10);
		cnt = (int)strtol(wx, NULL, 10);
		cnt = 10 * (cnt + k / 20) / k;
		mkfn_char(name, -1, c, cnt,
			(int)strtol(bb[0], NULL, 10), (int)strtol(bb[1], NULL, 10),
			(int)strtol(bb[2], NULL, 10), (int)strtol(bb[3], NULL, 10));
	}
	mkfn_header(fontname);
	for (;;) {
		if (fgets(line, 1024, stdin) == NULL)
			return 0;
		if (line[0] == '#')
			continue;
		if (strncmp(line, "StartKerningPairs", 17) == 0)
			break;
	}
	for (;;) {
		if (fgets(line, 1024, stdin) == NULL)
			return 0;
		if (line[0] == '#')
			continue;
		if (strncmp(line, "EndKernPairs", 12) == 0)
			return 0;
		if (sscanf(line, "KPX %s %s %s", lig1, lig2, wx) != 3)
			continue;
		k = 72000 / mkfn_res;
		cnt = (int)strtol(wx, NULL, 10);
		cnt = 10 * (cnt + k / 20) / k;
		mkfn_kern(lig1, lig2, cnt);
	}
	return 0;
}
