#include "lobject.h"
#include "Table.h"

int luaO_ceillog2 (unsigned int x);

static int countint (lua_Integer key, unsigned int *nums) {
  unsigned int k = ((lua_Unsigned)key - 1u < 0x80000000u) ? (unsigned int)key : 0;
  if (k != 0) {
    nums[luaO_ceillog2(k)]++;
    return 1;
  }
  else
    return 0;
}
