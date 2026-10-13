#include <stdio.h>
#include <string.h>

void def_piecesput(const char *name, const char *s1, const char *s2, const char *s3, const char *s4)
{
	int i;

	for (i = 0; i != 32; i++)
		if (strcmp(bracketpieces[i][0], name) == 0)
			break;
	if (i == 32) {
		for (i = 0; i != 32; i++)
			if (bracketpieces[i][0][0] == 0)
				break;
		if (i == 32)
			return;
	}
	snprintf(bracketpieces[i][0], 64, "%s", name);
	snprintf(bracketpieces[i][1], 64, "%s", s1);
	snprintf(bracketpieces[i][2], 64, "%s", s2);
	snprintf(bracketpieces[i][3], 64, "%s", s3);
	snprintf(bracketpieces[i][4], 64, "%s", s4);
}
