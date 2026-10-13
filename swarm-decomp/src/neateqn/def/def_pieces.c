#include <string.h>

void def_pieces(const char *name, const char **a, const char **b, const char **c, const char **d)
{
	int i;

	for (i = 0; i < 32; i++)
		if (strcmp(bracketpieces[i][0], name) == 0)
			break;
	if (i == 32)
		return;
	*a = bracketpieces[i][1][0] ? bracketpieces[i][1] : 0;
	*b = bracketpieces[i][2][0] ? bracketpieces[i][2] : 0;
	*c = bracketpieces[i][3][0] ? bracketpieces[i][3] : 0;
	*d = bracketpieces[i][4][0] ? bracketpieces[i][4] : 0;
}
