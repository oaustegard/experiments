#pragma once
/* a defined macro: name stored inline, body is a heap string */
struct macro {
	char name[32];		/* 0x00 */
	char *body;		/* 0x20 */
};
