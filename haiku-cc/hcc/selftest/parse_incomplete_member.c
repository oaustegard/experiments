// EXPECT-ERROR: [parse]
// A struct member must not have an incomplete struct type (C11 6.7.2.1p3).
struct inner;
struct outer { struct inner i; };
int main(void) { return 0; }
