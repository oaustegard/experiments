// EXPECT-ERROR: [parse]
// designated_init is a struct-only attribute (gcc PR c/59855): typedef of int must be rejected.
typedef int vvv __attribute__((designated_init));
