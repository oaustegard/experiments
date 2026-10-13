static IdxT choosePivot (IdxT lo, IdxT up, unsigned int rnd) {
  IdxT r4 = (up - lo) / 4;
  IdxT p = rnd % (r4 * 2) + (lo + r4);
  return p;
}
