// EXPECT-ERROR: 'inline' in empty declaration
// `inline` is a function specifier, not a type: a declaration with no declarator is an error.
int inline;
int main(void) { return 0; }
