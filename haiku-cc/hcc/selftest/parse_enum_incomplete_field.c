// EXPECT-ERROR: [parse]
// A forward-declared enum without a body or fixed underlying type is incomplete,
// so it cannot be the type of a struct member (or bit-field).
struct s { enum E e : 8; };
int main(void) { return 0; }
