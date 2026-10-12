// EXPECT-EXIT: 0
// Consecutive ## operators act as one ## (C11 6.10.3.3 / GCC behaviour).
#define cat(x, y) x##########y
#define cat3(x, y, z) x ## ## y ## z
int cat(ab, cd) = 1;
int cat3(a, b, c) = 2;

int abc = 2;
int main(void) { return 0; }
