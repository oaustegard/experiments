#include <stdio.h>
#include <string.h>
#include "fn_startswith.h"

static int
pn_proc(char *path)
{
	char *p;
	long start, mark;
	int hflag, i, f;

	if (path == NULL) {
		pn_fp = stdin;
	} else {
		pn_fp = fopen(path, "r");
		if (pn_fp == NULL)
			return 1;
	}
	pn_path = path;
	pn_pos = 0;
	pn_num = 0;
	pn_buf[0] = '\0';
	goto header;

putline:
	pn_rev = 0;
	p = pn_buf;
putl2:
	fputs(p, stdout);
	if (*p != '\n')
		goto body;
header:
	if (pn_rev)
		goto putline;
	pn_pos += strlen(pn_buf);
	p = fgets(pn_buf, 4096, pn_fp);
	if (p)
		goto putl2;
	if (path == NULL) {
		path = "stdin";
	} else {
		fclose(pn_fp);
	}
	fprintf(stderr, "%s: %d messages\n", path, pn_num);
	return 0;

body:
	start = pn_pos;
	hflag = 0;
bodyloop:
	if (!pn_rev)
		goto bread;
scanstart:
	pn_rev = 0;
	p = pn_buf;
scan:
	if (*p == 'F')
		goto hfrom;
	if (*p == '\n')
		goto eol;
	if (*p == ' ' || *p == '\t') {
		if (hflag)
			goto hprint;
		hflag = 0;
		goto nxt;
	}
hdrchk:
	for (i = 0; i < hdrs_n; i++)
		if (startswith(p, hdrs[i]))
			goto hprint;
	hflag = 0;
nxt:
	if (pn_rev)
		goto scanstart;
bread:
	pn_pos += strlen(pn_buf);
	p = fgets(pn_buf, 4096, pn_fp);
	if (p)
		goto scan;
	f = pn_rev;
	goto bodyend;
eol:
	pn_rev = 1;
	f = 1;
	goto bodyend;
hfrom:
	if (p[1] == 'r' && p[2] == 'o' && p[3] == 'm' && p[4] == ' ')
		goto eol;
	goto hdrchk;
hprint:
	fputs(p, stdout);
	hflag = 1;
	goto bodyloop;

bodyend:
	mark = pn_pos;
bodycond:
	if (f)
		goto bodyprint;
	pn_pos += strlen(pn_buf);
	p = fgets(pn_buf, 4096, pn_fp);
	if (p)
		goto bcheck;
	goto end;
bodyprint:
	pn_rev = 0;
	p = pn_buf;
bcheck:
	if (*p == 'F' && p[1] == 'r' && p[2] == 'o' && p[3] == 'm' && p[4] == ' ') {
		pn_rev = 1;
		goto end;
	}
	if (maxlen < 0 || pn_pos - mark + (long)strlen(p) < maxlen)
		fputs(p, stdout);
	f = pn_rev;
	goto bodycond;

end:
	if (pn_path && maxlen == 0)
		goto info;
check:
	if (maxlen < 0)
		goto next;
	if (pn_pos - mark >= maxlen)
		fputc('\n', stdout);
next:
	pn_num++;
	goto header;
info:
	printf("Neatmail-Source: %d@%s %ld %ld\n", pn_num, pn_path, start, pn_pos);
	goto check;
}
