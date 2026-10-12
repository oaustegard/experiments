/* { dg-do compile } */
/* { dg-additional-options "-ftree-vectorize" } */

char ff_rv60_idct16x16_add_dst_1, ff_rv60_idct16x16_add___trans_tmp_1;
short ff_rv60_idct16x16_add_tmp[56];
void ff_rv60_idct16x16_add() {
  for (int y; y; y++) {
    int tmp00 = 31 * ff_rv60_idct16x16_add_tmp[y * 16] -
                7 * ff_rv60_idct16x16_add_tmp[y * 16 + 1] -
                6 * ff_rv60_idct16x16_add_tmp[y * 16 + 10] -
                20 * ff_rv60_idct16x16_add_tmp[y * 16 + 4],
        tmp02 = 20 * ff_rv60_idct16x16_add_tmp[y * 16] -
                6 * ff_rv60_idct16x16_add_tmp[y * 16 + 1] +
                7 * ff_rv60_idct16x16_add_tmp[y * 16 + 10] +
                31 * ff_rv60_idct16x16_add_tmp[y * 16 + 4];
    ff_rv60_idct16x16_add_dst_1 = (tmp00 >> 7) + (tmp02 >> 7);
    {
      if (ff_rv60_idct16x16_add_dst_1)
        ff_rv60_idct16x16_add___trans_tmp_1 = ff_rv60_idct16x16_add_dst_1;
    }
    ff_rv60_idct16x16_add_dst_1 = ff_rv60_idct16x16_add___trans_tmp_1;
  }
}
