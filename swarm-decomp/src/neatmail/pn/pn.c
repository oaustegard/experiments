#include <stdio.h>
#include <stdlib.h>

int
pn(char **argv)
{
	char *bl[16] = {0};
	char *s;
	int i, nb, j, c;

	nb = 0;
	i = 0;
	if (argv[0] == NULL) {
		puts("usage: neatmail pn [options] [imbox] <imbox >ombox\n\noptions:\n   -h hdr  \tspecify headers to include\n   -H      \tinclude the default set of headers\n   -s size \tmaximum message body length to include");
		return 1;
	}
	for (;;) {
		s = argv[i];
		if (s == NULL)
			break;
		if (s[0] != '-') {
			for (; argv[i]; i++)
				bl[nb++] = argv[i];
			break;
		}
		i++;
		c = s[1];
		if (c == 'h') {
			if (s[2] == '\0')
				s = argv[i++];
			else
				s += 2;
			if (hdrs_n <= 127)
				hdrs[hdrs_n++] = s;
			continue;
		}
		if (c == 'H') {
			if (hdrs_n > 127)
				continue;
			hdrs[hdrs_n++] = "From:";
			if (hdrs_n == 128)
				continue;
			hdrs[hdrs_n++] = "To:";
			if (hdrs_n == 128)
				continue;
			hdrs[hdrs_n++] = "Cc:";
			if (hdrs_n == 128)
				continue;
			hdrs[hdrs_n++] = "Bcc:";
			if (hdrs_n == 128)
				continue;
			hdrs[hdrs_n++] = "Subject:";
			if (hdrs_n == 128)
				continue;
			hdrs[hdrs_n++] = "Date:";
			if (hdrs_n == 128)
				continue;
			hdrs[hdrs_n++] = "Message-ID:";
			if (hdrs_n == 128)
				continue;
			hdrs[hdrs_n++] = "Reply-To:";
			if (hdrs_n == 128)
				continue;
			hdrs[hdrs_n++] = "In-Reply-To:";
			if (hdrs_n == 128)
				continue;
			hdrs[hdrs_n++] = "References:";
			if (hdrs_n == 128)
				continue;
			hdrs[hdrs_n++] = "MIME-Version:";
			if (hdrs_n == 128)
				continue;
			hdrs[hdrs_n++] = "Content-Type:";
			if (hdrs_n == 128)
				continue;
			hdrs[hdrs_n++] = "Content-Transfer-Encoding:";
			if (hdrs_n == 128)
				continue;
			hdrs[hdrs_n++] = "User-Agent:";
			if (hdrs_n == 128)
				continue;
			hdrs[hdrs_n++] = "X-Mailer:";
			if (hdrs_n == 128)
				continue;
			hdrs[hdrs_n++] = "Organization:";
			if (hdrs_n == 128)
				continue;
			hdrs[hdrs_n++] = "Status:";
			if (hdrs_n == 128)
				continue;
			hdrs[hdrs_n++] = "Neatmail";
			continue;
		}
		if (c == 's') {
			if (s[2] != '\0')
				s += 2;
			else
				s = argv[i++];
			maxlen = strtol(s, NULL, 10);
			continue;
		}
		if (c == 'b') {
			if (s[2] == '\0')
				bl[nb++] = argv[i++];
			else
				bl[nb++] = s + 2;
			continue;
		}
	}
	if (nb == 0) {
		pn_proc(bl[nb]);
		return 0;
	}
	for (j = 0; j < nb; j++)
		if (pn_proc(bl[j]))
			fprintf(stderr, "pn: cannot open <%s>\n", bl[j]);
	return 0;
}
