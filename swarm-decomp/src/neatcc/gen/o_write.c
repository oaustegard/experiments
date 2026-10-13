void o_write(int fd)
{
	i_done();
	out_write(fd, mem_buf(&cs), mem_len(&cs), mem_buf(&ds), mem_len(&ds));
	free(loc_off);
	free(ds_name);
	free(ds_off);
	mem_done(&cs);
	mem_done(&ds);
}
