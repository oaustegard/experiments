#include "hyph_ext.h"
#include "hcode_strcpy.h"
#include "hy_cget.h"
#include "fn_memset.h"
#include "hy_dohyph.h"

static void hy_dohyph(char *s, char *w, int n)
{
	char buf[256] = {0};
	char nums[256] = {0};
	int offs[256];
	int pos[256] = {0};
	char tmp[32];
	size_t len;
	int last, nc, i, j, k, p, m, ch = 0, plen;

	hcode_strcpy(buf, w, pos, 1);
	len = strlen(buf);
	if (len <= 1) {
		memset(s, 0, len);
		return;
	}
	k = 0;
	do {
		offs[k] = ch;
		last = k;
		k++;
		ch += hy_cget(tmp, buf + ch);
	} while (ch < (int)len - 1);
	nc = last + 1;
	if (last == 0) {
		memset(s, 0, len);
		return;
	}
	for (i = 0; i <= last; i++) {
		plen = -1;
		while ((p = pref_prefix(hypref, buf + offs[i], &plen)) >= 0) {
			m = strlen(hypats + hyoff[p]);
			for (k = 0; k < m + 1; k++) {
				if (nums[offs[i] + k] < hynums[hyoff[p] + k])
					nums[offs[i] + k] = hynums[hyoff[p] + k];
			}
		}
	}
	memset(s, 0, len);
	if (nc <= 5)
		return;
	for (j = 3; j + 2 < nc; j++) {
		if (!(nums[offs[j]] & 1))
			continue;
		if (buf[offs[j - 1]] == '.' || buf[offs[j]] == '.' ||
		    buf[offs[j - 2]] == '.' || buf[offs[j + 1]] == '.')
			continue;
		if ((n & 4) && buf[offs[j + 2]] == '.')
			continue;
		if ((n & 8) && buf[offs[j - 3]] == '.')
			continue;
		s[pos[offs[j]]] = 1;
	}
}
