#pragma once
/* input source level: a file, or a macro expansion pushed by src_push */
struct esrc {
	struct esrc *prev;	/* 0x00 */
	char *text;		/* 0x08 */
	int pos;		/* 0x10 */
	int back[1000];		/* 0x14: pushed-back characters */
	int nback;		/* 0xfb4 */
	char *arg[10];		/* 0xfb8: macro arguments */
	int nargs;		/* 0x1008: nonzero if arguments were given */
};
