// EXPECT-EXIT: 0
// EXPECT-FLAGS: -fgnu89-inline
// -fgnu89-inline defines __GNUC_GNU_INLINE__ and not __GNUC_STDC_INLINE__.
#ifndef __GNUC_GNU_INLINE__
#error __GNUC_GNU_INLINE__ is not defined
#endif
#ifdef __GNUC_STDC_INLINE__
#error __GNUC_STDC_INLINE__ is defined
#endif
int main(void) { return 0; }
