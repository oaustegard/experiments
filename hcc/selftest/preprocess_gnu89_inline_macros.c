// EXPECT-EXIT: 0
// Default mode defines __GNUC_STDC_INLINE__ and not __GNUC_GNU_INLINE__.
#ifndef __GNUC_STDC_INLINE__
#error stdc inline macro missing
#endif
#ifdef __GNUC_GNU_INLINE__
#error gnu inline macro should not be defined
#endif
int main(void) { return 0; }
