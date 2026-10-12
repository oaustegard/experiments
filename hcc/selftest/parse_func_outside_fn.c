// EXPECT-EXIT: 0
// __FUNCTION__/__PRETTY_FUNCTION__/__func__ outside a function are empty strings;
// C23 *_WIDTH__ macros are predefined.
const char *a = __func__;
const char *b = __FUNCTION__;
const char *c = __PRETTY_FUNCTION__;
int w = __SCHAR_WIDTH__ + __INT_LEAST16_WIDTH__;
int main(void) {
    return a[0] + b[0] + c[0] + (w == 24 ? 0 : 1);
}
