// EXPECT-EXIT: 0
#include <string.h>
int main(void) {
    const char *s = __TIMESTAMP__;
    if (strlen(s) != 24) return 1;
    if (s[10] != ' ' && s[8] != ' ') return 2;
    return 0;
}
