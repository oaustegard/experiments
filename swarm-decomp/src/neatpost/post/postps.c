#include "outps.h"

static void postps(void)
{
	char word[2048];
	char line[1008];
	char arg[128];
	char *p;
	int *ip;
	int c, n, sz, r, n1, n2;

	nextword(word);
	p = line;
	c = getc(stdin);
	while (c > 0 && c != '\n') {
		*p++ = c;
		c = getc(stdin);
	}
	if (c == '\n')
		ungetc(c, stdin);
	*p = 0;
	if (strcmp(word, "PS") == 0) {
		out("%s\n", line);
	} else if (strcmp(word, "ps") == 0) {
		out("%s\n", line);
	} else if (strcmp(word, "rotate") == 0) {
		outrotate(strtol(line, NULL, 10));
	} else if (strcmp(word, "eps") == 0 || strcmp(word, "pdf") == 0) {
		r = sscanf(strcut(arg, line), "%d %d", &n1, &n2);
		if (r <= 0) {
			n1 = 0;
			n2 = 0;
		} else if (r == 1) {
			n2 = 0;
		}
		if (arg[0]) {
			if (strcmp(word, "eps") == 0)
				outeps(arg, n1, n2);
			else if (strcmp(word, "pdf") == 0)
				outpdf(arg, n1, n2);
		}
	} else if (strcmp(word, "name") == 0) {
		n = name_n;
		if (n == name_sz) {
			sz = n ? 2 * n : 128;
			name_sz = sz;
			p = malloc(sz * 64);
			memcpy(p, name_desc, n * 64);
			memset(p + n * 64, 0, (sz - n) * 64);
			free(name_desc);
			name_desc = p;
			ip = malloc(sz * 4);
			memcpy(ip, name_page, n * 4);
			memset(ip + n, 0, (sz - n) * 4);
			free(name_page);
			name_page = ip;
			ip = malloc(sz * 4);
			memcpy(ip, name_offset, n * 4);
			memset(ip + n, 0, (sz - n) * 4);
			free(name_offset);
			name_offset = ip;
		}
		p = name_desc + n * 64;
		r = sscanf(strcut(p, line), "%d %d", name_page + n, name_offset + n);
		if (*p && r > 0)
			name_n = n + 1;
	} else if (strcmp(word, "mark") == 0) {
		n = mark_n;
		if (n == mark_sz) {
			sz = n ? 2 * n : 128;
			mark_sz = sz;
			p = malloc(sz * 256);
			memcpy(p, mark_desc, n * 256);
			memset(p + n * 256, 0, (sz - n) * 256);
			free(mark_desc);
			mark_desc = p;
			ip = malloc(sz * 4);
			memcpy(ip, mark_page, n * 4);
			memset(ip + n, 0, (sz - n) * 4);
			free(mark_page);
			mark_page = ip;
			ip = malloc(sz * 4);
			memcpy(ip, mark_offset, n * 4);
			memset(ip + n, 0, (sz - n) * 4);
			free(mark_offset);
			mark_offset = ip;
			ip = malloc(sz * 4);
			memcpy(ip, mark_level, n * 4);
			memset(ip + n, 0, (sz - n) * 4);
			free(mark_level);
			mark_level = ip;
		}
		p = mark_desc + n * 256;
		r = sscanf(strcut(p, line), "%d %d %d", mark_page + n, mark_offset + n, mark_level + n);
		if (*p && r > 0)
			mark_n = n + 1;
	} else if (strcmp(word, "link") == 0) {
		r = sscanf(strcut(arg, line), "%d %d", &n1, &n2);
		if (arg[0] && r == 2)
			outlink(arg, n1, n2);
	} else if (strcmp(word, "info") == 0) {
		p = line;
		while (*p == ' ')
			p++;
		n = 0;
		while (*p && *p != ' ') {
			if (n <= 126)
				arg[n++] = *p;
			p++;
		}
		arg[n] = 0;
		while (*p == ' ')
			p++;
		outinfo(arg);
	} else if (strcmp(word, "set") == 0) {
		char key[128];

		r = sscanf(line, "%127s %127s", key, arg);
		if (r == 2)
			outset(key, arg);
	} else if (strcmp(word, "BeginObject") == 0) {
		drawmbeg(line);
	} else if (strcmp(word, "EndObject") == 0) {
		drawmend(line);
	}
}
