// EXPECT-ERROR: [parse]
// File-scope object of incomplete struct type with no initializer is an error.
typedef struct foo_ foo_t;
foo_t bar;
int main(void) { return 0; }
