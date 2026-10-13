#pragma once

/* offsets from sbuf_buf / sbuf_done / sbuf_len listings */
struct sbuf {
	char *buf;	/* 0x0 */
	int len;	/* 0x8 */
	int size;	/* 0xc */
};
