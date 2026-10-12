// EXPECT-EXIT: 0
// EXPECT-FLAGS: -std=c2x
// C23: a constexpr object is an integer constant, so it may size a bit-field.
constexpr int w = 3;
struct s { int x : w; };
int main(void) { struct s v; return sizeof v != sizeof(int); }
