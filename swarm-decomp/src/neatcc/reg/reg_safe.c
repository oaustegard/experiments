int reg_safe(long i)
{
	int r = 0;

	if (loc_n > i)
		r = loc_ptr[i] == 0;
	return r;
}
