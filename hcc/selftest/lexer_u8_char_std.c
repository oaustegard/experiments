// EXPECT-EXIT: 0
// u8 is an identifier before C23, so a macro named u8 expands here (C11 default).
#define u8 z(
#define z(x) 0
unsigned char a = u8'a');
int main(void) { return a; }
