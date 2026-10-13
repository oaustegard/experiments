#pragma once

/* one entry of the troff request table (tr/cmds.c) */
struct trcmd {
	char *name;
	void (*fn)();
	void (*mk)();
};
