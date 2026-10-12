// EXPECT-EXIT: 0
// #include <NAME with NAME macro-expanded when the closing '>' is not on the line.
#define hdr <stddef.h>
#define name stddef.h>
#include <name
int main(void) { size_t n = 0; return (int)n; }
