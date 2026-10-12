// EXPECT-FLAGS: -std=gnu23
// EXPECT-ERROR: declared in underspecified object initializer
// C23: a tag may not be declared inside the initializer of an `auto` object.
void f(void) {
    auto x = ({ struct s; struct s *p = 0; p; });
}
int main(void) { return 0; }
