void i_done(void)
{
	free(jmp_off);
	free(jmp_dst);
	free(jmp_op);
	free(lab_loc);
}
