// EXPECT-EXIT: 0
// ms_struct: a zero-width bit-field after a bit-field pads to its type and
// aligns the struct; a following zero-width bit-field does not pad again.
struct S { char foo : 4; short : 0; char bar; } __attribute__((__ms_struct__));
struct T { char foo : 4; short : 0; long : 0; char bar; } __attribute__((__ms_struct__));
int a1[(sizeof(struct S) == 4) ? 1 : -1];
int a2[(sizeof(struct T) == 4) ? 1 : -1];
int main(void) { return 0; }
