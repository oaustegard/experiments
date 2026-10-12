// EXPECT-EXIT: 0
// hcc does not implement full 128-bit scalar arithmetic (values are one 64-bit
// word), so it must not advertise __SIZEOF_INT128__: libraries such as QuickJS
// then take their portable limb path instead of the broken 128-bit one.
#ifdef __SIZEOF_INT128__
#error __SIZEOF_INT128__ must not be defined
#endif
int main(void) {
    return 0;
}
