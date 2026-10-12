SELECT ieee754(0.5), ieee754(-1.25), ieee754(1e100), ieee754(0.1), ieee754_mantissa(0.5), ieee754_exponent(8.0);
SELECT ieee754_to_blob(1.0) IS NOT NULL, hex(ieee754_to_blob(-2.5)), ieee754_from_blob(ieee754_to_blob(3.14159)), ieee754(5, 10), ieee754(1, -1074);
SELECT ieee754_from_blob(x'3ff0000000000000'), ieee754_from_blob(x'7ff0000000000000'), ieee754(3), ieee754(NULL);
