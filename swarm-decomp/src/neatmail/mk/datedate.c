#include <stdlib.h>
#include <string.h>
#include <time.h>

static int datedate(char *s) {
	struct tm tm = {0};
	char buf[128];
	char *p;
	long t;
	int tz, i;

	p = readtok((unsigned char *)s, buf);
	if (strchr(buf, ','))
		p = readtok((unsigned char *)p, buf);
	tm.tm_mday = strtol(buf, NULL, 10);
	p = readtok((unsigned char *)p, buf);
	for (i = 0; i < 12; i++)
		if (strcmp(months[i], buf) == 0)
			tm.tm_mon = i;
	p = readtok((unsigned char *)p, buf);
	tm.tm_year = strtol(buf, NULL, 10) - 1900;
	p = readtok((unsigned char *)p, buf);
	tm.tm_hour = strtol(buf, NULL, 10);
	p = readtok((unsigned char *)p, buf);
	tm.tm_min = strtol(buf, NULL, 10);
	p = readtok((unsigned char *)p, buf);
	if (buf[0] == '+' || buf[0] == '-') {
		;
	} else {
		tm.tm_sec = strtol(buf, NULL, 10);
		readtok((unsigned char *)p, buf);
	}
	tz = strtol(buf, NULL, 10);
	t = mktime(&tm);
	if (tz >= 0)
		return t - (tz / 100 * 3600 + tz % 100 * 60);
	else
		return t + (-tz / 100 * 3600 + -tz % 100 * 60);
}
