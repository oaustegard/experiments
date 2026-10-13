// EXPECT-EXIT: 1
// GNU __auto_type keeps the qualifiers written with it (const object, then extern redecl).
const __auto_type ll = 1LL;
extern const long long ll;
int main(void) { return (int) ll; }
