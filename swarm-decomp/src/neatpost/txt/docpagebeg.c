void docpagebeg(int n)
{
	memset(o_pg, 0, p_cwd * p_cht);
	if (n > 1)
		out("\f\n");
}
