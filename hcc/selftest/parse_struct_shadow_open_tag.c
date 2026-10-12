// EXPECT-EXIT: 7
// A struct tag whose body is still open in an enclosing scope is shadowed by a
// new definition of the same tag inside a statement expression (not an error).
int main(void)
{
    struct s { int a; typeof(({ struct s { char c[3]; } t; t; })) *p; } v;
    return sizeof(v.a) + sizeof(*v.p);
}
