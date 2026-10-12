// EXPECT-EXIT: 0
// #pragma weak NAME = TARGET makes NAME a weak alias of a static variable.
static int target = 42;
#pragma weak alias_v = target
extern int alias_v;
int main(void) {
    return alias_v == 42 ? 0 : 1;
}
