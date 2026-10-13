void i_wrap(long unused, long regs, long fsz, int va, long save, long locals)
{
	int sv;
	int cnt;
	int i;
	int *p;
	long codelen;
	long start;
	char *body;

	if (jmp_ret + 5 == mem_len(&cs)) {
		mem_cut(&cs, jmp_ret);
		jmp_n--;
	}
	lab_add(0);
	codelen = mem_len(&cs);
	body = mem_get(&cs);
	if (regs) {
		mem_put(&cs, "X", 1);
		p = argregs + 5;
		do {
			if (regs & (1 << *p))
				op_x(0x50 | (*p & 7), 0, *p, 8);
		} while (p-- != argregs);
		mem_put(&cs, "P", 1);
	}
	if (va) {
		mem_put(&cs, "U", 1);
		mem_put(&cs, "H\x89\xe5", 3);
	}
	cnt = 0;
	for (p = argregs; p != argregs + 6; p++)
		if (regs & (1 << *p))
			cnt++;
	sv = cnt * 8;
	if (fsz) {
		fsz += 16 - (sv + fsz) % 16;
		if (!save) {
			i_subsp(fsz);
			start = mem_len(&cs);
			mem_put(&cs, body, codelen);
			free(body);
			goto done;
		}
		cnt = 0;
		for (i = 0; i < 16; i++)
			if (((1 << i) & 0xffcf) & save)
				cnt++;
		i_subsp(-locals - cnt * 8);
		locals += fsz;
	} else if (!save) {
		start = mem_len(&cs);
		mem_put(&cs, body, codelen);
		free(body);
		goto done;
	}
	for (i = 0; i < 16; i++)
		if (((1 << i) & 0xffcf) & save)
			op_x(0x50 | (i & 7), 0, i, 8);
	if (locals) {
		i_subsp(locals);
		start = mem_len(&cs);
		mem_put(&cs, body, codelen);
		free(body);
		i_subsp(-locals);
	} else {
		start = mem_len(&cs);
		mem_put(&cs, body, codelen);
		free(body);
	}
	for (i = 15; i >= 0; i--)
		if (((1 << i) & 0xffcf) & save)
			op_x(0x58 | (i & 7), 0, i, 8);
done:
	if (va)
		mem_put(&cs, "\xc9", 1);
	if (regs) {
		mem_put(&cs, "\xc2", 1);
		op_put(sv, 2);
	} else {
		mem_put(&cs, "\xc3", 1);
	}
	for (i = 0; i < rel_n; i++)
		rel_off[i] += start;
	for (i = 0; i < jmp_n; i++)
		jmp_off[i] += start;
	for (i = 0; i < lab_sz; i++)
		lab_loc[i] += start;
}
