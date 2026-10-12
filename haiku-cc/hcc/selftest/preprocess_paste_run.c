// EXPECT-EXIT: 0
// Consecutive `##` operators act as one: x##########y pastes to abcd.
#define cat(x,y) x##########y
int abcd = 0;
int main(void) { int *p = &cat(ab,cd); return p != &abcd; }
