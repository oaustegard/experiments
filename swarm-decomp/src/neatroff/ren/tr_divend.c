void tr_divend(void)
{
	if (ren_div <= 0)
		errdie("neatroff: diversion stack empty\n");
	odiv_end();
	ren_div--;
}
