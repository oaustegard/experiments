#pragma once
#include "type.h"
/* one entry of the arrays table; 24 bytes: a type, then an int size at +20 */
struct array {
	struct type t;	/* +0 */
	int size;	/* +20 */
};
