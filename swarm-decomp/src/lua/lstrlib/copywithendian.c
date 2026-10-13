static const union {
  int dummy;
  char little;
} nativeendian = {1};

static void copywithendian (char *dest, const char *src, int size, int islittle) {
  if (islittle == nativeendian.little)
    memcpy(dest, src, size);
  else {
    dest += size - 1;
    while (size-- != 0)
      *(dest--) = *(src++);
  }
}
