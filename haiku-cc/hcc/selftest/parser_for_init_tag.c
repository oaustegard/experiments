// EXPECT-ERROR: declared in 'for' loop initial declaration
// A struct tag may not be defined in a for-statement init declaration.
int main(void) {
    for (struct s { int p; } *p = 0; ;)
        ;
    return 0;
}
