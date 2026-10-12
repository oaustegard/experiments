// EXPECT-EXIT: 0
// File-scope __auto_type takes the type of its initializer.
__auto_type g = 10U;
extern unsigned int g;
static __auto_type c = (char) 1;
int main(void) {
    if (g != 10U || c != 1)
        return 1;
    return 0;
}
