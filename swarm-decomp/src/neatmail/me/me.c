#include <ctype.h>
#include <stdio.h>

int me(char **argv)
{
	char buf[1024];
	char c;
	unsigned char pc;
	char *p;
	int ch, i, mime;
	unsigned short nl;

	nl = 0x200a;
L18:
	if (!fgets(buf, 1024, stdin))
		goto L197;
	c = buf[0];
	if (c != ' ' && c != '\t')
		out_mhdr = 0;
	i = 0;
	while ((pc = "from:"[i]) != 0) {
		if (tolower((unsigned char)buf[i]) != tolower(pc))
			break;
		i++;
	}
	if (!pc)
		goto L81;
	i = 0;
L320:
	i = 0;
L341:
	pc = "to:"[i];
	if (!pc)
		goto L81;
	if (tolower((unsigned char)buf[i]) != tolower(pc))
		goto L380;
	i++;
	goto L341;
L380:
	i = 0;
L3a1:
	pc = "cc:"[i];
	if (!pc)
		goto L81;
	if (tolower((unsigned char)buf[i]) != tolower(pc))
		goto L3b2;
	i++;
	goto L3a1;
L3b2:
	i = 0;
L3dd:
	pc = "bcc:"[i];
	if (!pc)
		goto L81;
	if (tolower((unsigned char)buf[i]) != tolower(pc))
		goto L8b;
	i++;
	goto L3dd;
L81:
	out_mhdr = 'a';
L8b:
	i = 0;
La9:
	pc = "subject:"[i];
	if (!pc)
		goto Lb5;
	if (tolower((unsigned char)buf[i]) != tolower(pc))
		goto Lbf;
	i++;
	goto La9;
Lb5:
	out_mhdr = 's';
Lbf:
	if (!c)
		goto L18;
	p = buf;
	goto L111;
L111:
	mime = out_mime;
	p++;
	ch = (unsigned char)c;
	if (mime)
		goto Ld0;
	if (c != '\n')
		goto L1da;
	out[out_n] = '\n';
	out_n++;
	goto L144;
L144:
	out[out_n] = 0;
	fputs(out, stdout);
	c = *p;
	out_n = 0;
	if (c)
		goto L111;
	goto L170;
L170:
	if (buf[0] == '\n')
		goto L197;
	if (buf[0] == '\r')
		goto L197;
	goto L18;
L197:
	while (fgets(buf, 1024, stdin))
		fputs(buf, stdout);
	return 0;
L101:
	q_put(ch);
L109:
	c = *p;
	if (!c)
		goto L170;
	goto L111;
Ld0:
	if (ch == 13)
		goto L1c0;
	if (ch == 10)
		goto L1c0;
	if (out_mhdr == 'a')
		goto L278;
	goto Lf1;
Lf1:
	if (out_n > 65)
		goto L250;
	goto L101;
L1c0:
	q_end();
	mime = out_mime;
	if (c == '\n')
		goto L1e2;
	goto L1da;
L1da:
	if (out_n > 75)
		goto L1e2;
	if (out_mhdr != 0 && c < 0)
		goto L358;
L223:
	if (mime)
		goto L101;
	out[out_n] = c;
	out_n++;
	goto L109;
L1e2:
	out[out_n] = c;
	out_n++;
	if (mime)
		q_end();
	goto L144;
L1fa:
	q_end();
	goto L144;
L358:
	if (mime)
		goto L101;
	q_beg();
	mime = out_mime;
	goto L223;
L278:
	if (c == ' ')
		goto L2b0;
	if (c < 0)
		goto Lf1;
	q_end();
	mime = out_mime;
	if (out_n > 75)
		goto L1e2;
	goto L223;
L2b0:
	if (out_n <= 65)
		goto L101;
	goto L2bd;
L250:
	if ((ch & 0xc0) != 0x80)
		goto L2bd;
	if (out_n <= 75)
		goto L101;
	out[out_n] = c;
	out_n++;
	goto L1fa;
L2bd:
	q_end();
	mime = out_mime;
	if (mime)
		q_end();
	out[out_n] = 0;
	fputs(out, stdout);
	out[0] = nl;
	out_n = 2;
	q_beg();
	mime = out_mime;
	goto L1da;
}
