#pragma once
struct mem {
	char *buf;
	long len;
	long cap;
};
char *mem_buf(struct mem *m);
