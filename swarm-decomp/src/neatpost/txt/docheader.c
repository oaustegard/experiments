void docheader(int unused, int pw, int ph)
{
	c_wd = c_wdpt * dev_res / 72;
	c_ht = c_htpt * dev_res / 72;
	p_cwd = pw * dev_res / 254 / c_wd;
	p_cht = ph * dev_res / 254 / c_ht;
	o_pg = malloc(p_cwd * p_cht);
}
