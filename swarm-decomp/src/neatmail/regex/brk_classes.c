#include "brk_class.h"

static const struct brk_class brk_classes[11] = {
	{ ":alnum:", "a-zA-Z0-9" },
	{ ":alpha:", "a-zA-Z" },
	{ ":blank:", " \t" },
	{ ":digit:", "0-9" },
	{ ":lower:", "a-z" },
	{ ":print:", " -~" },
	{ ":punct:", "][!\"#$%&'()*+,./:;<=>?@\\^_`{|}~-" },
	{ ":space:", " \t\r\n\x0b\x0c" },
	{ ":upper:", "A-Z" },
	{ ":word:", "a-zA-Z0-9_" },
	{ ":xdigit:", "a-fA-F0-9" },
};
