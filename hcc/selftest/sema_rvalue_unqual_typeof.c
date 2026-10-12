// EXPECT-EXIT: 0
// The result of unary +/-, a comma or a conditional is an rvalue: its type drops qualifiers.
const int jc;
extern typeof(0, jc) j;
extern int j;
extern typeof(+jc) k;
extern int k;
extern typeof(-jc) m;
extern int m;
extern typeof(1 ? jc : 0) n;
extern int n;
int main(void) { return 0; }
