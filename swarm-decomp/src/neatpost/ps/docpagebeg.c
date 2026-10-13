#include "ps_extern.h"

void docpagebeg(int n)
{
	out("%%%%Page: %d %d\n", n, n);
	out("/saveobj save def\n");
	out("mark\n");
	out("%d pagesetup\n", n);
}
