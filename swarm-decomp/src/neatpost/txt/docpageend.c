void docpageend(void)
{
	int r, x, bottom, right, c;

	bottom = 0;
	for (r = 0; r < p_cht; r++)
		for (x = 0; x < p_cwd; x++)
			if (((char *)o_pg)[r * p_cwd + x])
				bottom = r + 1;
	for (r = 0; r < bottom; r++) {
		right = 0;
		for (x = 0; x < p_cwd; x++)
			if (((char *)o_pg)[r * p_cwd + x])
				right = x + 1;
		for (x = 0; x < right; x++) {
			c = ((char *)o_pg)[r * p_cwd + x];
			__builtin_putchar(c ? c : ' ');
		}
		__builtin_putchar('\n');
	}
}
