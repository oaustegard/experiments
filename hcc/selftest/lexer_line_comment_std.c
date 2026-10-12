// EXPECT-EXIT: 0
// Line comments are still recognised in the default (gnu11) mode; the
// C89 `//` split is only applied when -std=c89/c90 is given.
#define h(x) #x
#define s(x) h(x)
#define foo 1 // trailing comment
int main(void) {
    int a = foo; // comment
    return (s(foo)[0] == '1' && a == 1) ? 0 : 1;
}
