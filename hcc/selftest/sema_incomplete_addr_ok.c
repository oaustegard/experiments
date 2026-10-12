// EXPECT-EXIT: 0
// Taking the address of an incomplete struct object is fine; the struct is completed later.
struct s;
extern struct s vs;
struct s *f(void) { return &vs; }
struct s { int m; };
struct s vs = { 3 };
int main(void) { return f()->m - 3; }
