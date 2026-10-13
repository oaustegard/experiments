#pragma once

/* one input level of the troff input stack (in_push allocates 0x4b8 bytes) */
struct src {
	int nil;
	char pad004[0x400 - 4];
	void *fp;		/* 0x400: FILE* being read (stdin is never closed), NULL for macros */
	char *name;		/* 0x408: copy of the source name */
	char **args;		/* 0x410: 32 macro argument strings, NULL when none */
	int back[32];		/* 0x418: characters pushed back by in_back */
	int nback;		/* 0x498: number of pushed-back characters */
	int pos;		/* 0x49c: read index into name when fp is NULL */
	int len;		/* 0x4a0: strlen of the name */
	int last;		/* 0x4a4: previous character read, to count lines */
	int lnum;		/* 0x4a8: current input line number */
	char pad4ac[0x4b0 - 0x4ac];
	struct src *prev;	/* 0x4b0: enclosing input level */
};
