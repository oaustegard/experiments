// EXPECT-EXIT: 0
#ifndef __PRAGMA_REDEFINE_EXTNAME
#error "__PRAGMA_REDEFINE_EXTNAME must be predefined"
#endif
int main(void) { return 0; }
