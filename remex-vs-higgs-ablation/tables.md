## Headline: remex vs HIGGS-like, matched actual bytes

recall@10 vs fp32 exact search, mean over 5 rotation seeds (worst seed in parentheses).

**arxiv768 — cosine**

| bits | B/vec remex | B/vec HIGGS-like | remex R@10 | HIGGS-like R@10 | delta | remex rho | HIGGS-like rho |
|---|---|---|---|---|---|---|---|
| 1 | 100 | 108 | 0.682 (0.677) | 0.688 (0.677) | +0.006 | 0.9063 | 0.9118 |
| 2 | 196 | 204 | 0.828 (0.812) | 0.845 (0.837) | +0.017 | 0.9733 | 0.9791 |
| 3 | 292 | 300 | 0.903 (0.896) | 0.912 (0.906) | +0.009 | 0.9922 | 0.9939 |
| 4 | 388 | 396 | 0.942 (0.937) | 0.948 (0.943) | +0.006 | 0.9978 | 0.9983 |
| 6 | 580 | 588 | 0.982 (0.979) | 0.985 (0.983) | +0.003 | 0.9998 | 0.9998 |
| 8 | 772 | 780 | 0.995 (0.993) | 0.993 (0.992) | -0.001 | 1.0000 | 1.0000 |

**arxiv768 — inner product**

| bits | B/vec remex | B/vec HIGGS-like | remex R@10 | HIGGS-like R@10 | delta | remex rho | HIGGS-like rho |
|---|---|---|---|---|---|---|---|
| 1 | 100 | 108 | 0.683 (0.674) | 0.668 (0.661) | -0.015 | 0.9059 | 0.9063 |
| 2 | 196 | 204 | 0.768 (0.753) | 0.788 (0.773) | +0.020 | 0.9597 | 0.9686 |
| 3 | 292 | 300 | 0.847 (0.839) | 0.869 (0.855) | +0.022 | 0.9847 | 0.9895 |
| 4 | 388 | 396 | 0.909 (0.899) | 0.924 (0.915) | +0.015 | 0.9951 | 0.9970 |
| 6 | 580 | 588 | 0.968 (0.959) | 0.974 (0.973) | +0.006 | 0.9996 | 0.9997 |
| 8 | 772 | 780 | 0.987 (0.984) | 0.991 (0.989) | +0.005 | 1.0000 | 1.0000 |

**glove100 — cosine**

| bits | B/vec remex | B/vec HIGGS-like | remex R@10 | HIGGS-like R@10 | delta | remex rho | HIGGS-like rho |
|---|---|---|---|---|---|---|---|
| 1 | 16 | 16 | 0.315 (0.313) | 0.340 (0.334) | +0.025 | 0.8023 | 0.8212 |
| 2 | 29 | 29 | 0.598 (0.596) | 0.636 (0.633) | +0.038 | 0.9412 | 0.9532 |
| 3 | 42 | 42 | 0.774 (0.772) | 0.804 (0.797) | +0.030 | 0.9832 | 0.9879 |
| 4 | 54 | 60 | 0.876 (0.872) | 0.893 (0.890) | +0.017 | 0.9954 | 0.9967 |
| 6 | 79 | 79 | 0.965 (0.963) | 0.967 (0.964) | +0.002 | 0.9997 | 0.9997 |
| 8 | 104 | 104 | 0.990 (0.990) | 0.990 (0.989) | -0.000 | 1.0000 | 1.0000 |

**glove100 — inner product**

| bits | B/vec remex | B/vec HIGGS-like | remex R@10 | HIGGS-like R@10 | delta | remex rho | HIGGS-like rho |
|---|---|---|---|---|---|---|---|
| 1 | 16 | 16 | 0.324 (0.319) | 0.348 (0.344) | +0.024 | 0.7886 | 0.8079 |
| 2 | 29 | 29 | 0.575 (0.567) | 0.614 (0.612) | +0.039 | 0.9341 | 0.9476 |
| 3 | 42 | 42 | 0.748 (0.743) | 0.793 (0.791) | +0.045 | 0.9806 | 0.9863 |
| 4 | 54 | 60 | 0.856 (0.854) | 0.884 (0.883) | +0.028 | 0.9947 | 0.9963 |
| 6 | 79 | 79 | 0.957 (0.955) | 0.963 (0.961) | +0.006 | 0.9996 | 0.9997 |
| 8 | 104 | 104 | 0.987 (0.986) | 0.989 (0.989) | +0.002 | 1.0000 | 1.0000 |

**nfcorpus1024 — cosine**

| bits | B/vec remex | B/vec HIGGS-like | remex R@10 | HIGGS-like R@10 | delta | remex rho | HIGGS-like rho |
|---|---|---|---|---|---|---|---|
| 1 | 132 | 144 | 0.657 (0.650) | 0.674 (0.658) | +0.017 | 0.9182 | 0.9240 |
| 2 | 260 | 272 | 0.814 (0.807) | 0.834 (0.829) | +0.020 | 0.9778 | 0.9822 |
| 3 | 388 | 400 | 0.896 (0.891) | 0.907 (0.902) | +0.011 | 0.9936 | 0.9949 |
| 4 | 516 | 528 | 0.941 (0.936) | 0.948 (0.946) | +0.006 | 0.9982 | 0.9986 |
| 6 | 772 | 784 | 0.980 (0.978) | 0.982 (0.982) | +0.002 | 0.9999 | 0.9999 |
| 8 | 1028 | 1040 | 0.994 (0.994) | 0.995 (0.994) | +0.001 | 1.0000 | 1.0000 |

**nfcorpus1024 — inner product**

| bits | B/vec remex | B/vec HIGGS-like | remex R@10 | HIGGS-like R@10 | delta | remex rho | HIGGS-like rho |
|---|---|---|---|---|---|---|---|
| 1 | 132 | 144 | 0.652 (0.644) | 0.646 (0.642) | -0.006 | 0.9290 | 0.9295 |
| 2 | 260 | 272 | 0.762 (0.759) | 0.785 (0.778) | +0.023 | 0.9733 | 0.9790 |
| 3 | 388 | 400 | 0.847 (0.841) | 0.875 (0.873) | +0.029 | 0.9903 | 0.9933 |
| 4 | 516 | 528 | 0.912 (0.908) | 0.925 (0.921) | +0.014 | 0.9971 | 0.9981 |
| 6 | 772 | 784 | 0.973 (0.969) | 0.975 (0.973) | +0.002 | 0.9998 | 0.9998 |
| 8 | 1028 | 1040 | 0.992 (0.990) | 0.993 (0.992) | +0.001 | 1.0000 | 1.0000 |

**fmnist784 — cosine**

| bits | B/vec remex | B/vec HIGGS-like | remex R@10 | HIGGS-like R@10 | delta | remex rho | HIGGS-like rho |
|---|---|---|---|---|---|---|---|
| 1 | 102 | 112 | 0.671 (0.662) | 0.650 (0.645) | -0.021 | 0.9887 | 0.9895 |
| 2 | 200 | 210 | 0.793 (0.790) | 0.796 (0.793) | +0.003 | 0.9970 | 0.9976 |
| 3 | 298 | 308 | 0.874 (0.873) | 0.883 (0.881) | +0.009 | 0.9991 | 0.9993 |
| 4 | 396 | 406 | 0.927 (0.924) | 0.933 (0.930) | +0.007 | 0.9998 | 0.9998 |
| 6 | 592 | 602 | 0.977 (0.977) | 0.980 (0.978) | +0.002 | 1.0000 | 1.0000 |
| 8 | 788 | 798 | 0.994 (0.993) | 0.994 (0.994) | +0.001 | 1.0000 | 1.0000 |

**fmnist784 — inner product**

| bits | B/vec remex | B/vec HIGGS-like | remex R@10 | HIGGS-like R@10 | delta | remex rho | HIGGS-like rho |
|---|---|---|---|---|---|---|---|
| 1 | 102 | 112 | 0.655 (0.631) | 0.608 (0.578) | -0.047 | 0.9968 | 0.9968 |
| 2 | 200 | 210 | 0.691 (0.675) | 0.710 (0.701) | +0.019 | 0.9988 | 0.9991 |
| 3 | 298 | 308 | 0.771 (0.749) | 0.808 (0.798) | +0.036 | 0.9996 | 0.9997 |
| 4 | 396 | 406 | 0.864 (0.861) | 0.884 (0.868) | +0.019 | 0.9999 | 0.9999 |
| 6 | 592 | 602 | 0.950 (0.942) | 0.966 (0.964) | +0.016 | 1.0000 | 1.0000 |
| 8 | 788 | 798 | 0.985 (0.983) | 0.989 (0.987) | +0.004 | 1.0000 | 1.0000 |

## Marginal effect of each axis

Mean change in recall@10 from flipping one axis with the other two held fixed, averaged over the 4 cells that differ only in that axis, +/- 1 sd across those cells. Positive = the HIGGS-lineage setting wins.

**arxiv768 — cosine**

| bits | A rotation: haar -> rht | B norm: exact -> blockscale | C codebook: scalar -> vector |
|---|---|---|---|
| 1 | -0.0020 ± 0.0013 | +0.0013 ± 0.0015 | +0.0056 ± 0.0012 |
| 2 | +0.0006 ± 0.0038 | -0.0002 ± 0.0032 | +0.0156 ± 0.0024 |
| 3 | -0.0040 ± 0.0026 | -0.0000 ± 0.0018 | +0.0128 ± 0.0032 |
| 4 | -0.0002 ± 0.0013 | -0.0002 ± 0.0014 | +0.0056 ± 0.0016 |
| 6 | +0.0009 ± 0.0010 | +0.0011 ± 0.0011 | +0.0016 ± 0.0008 |
| 8 | -0.0014 ± 0.0009 | -0.0002 ± 0.0013 | -0.0003 ± 0.0011 |

**arxiv768 — inner product**

| bits | A rotation: haar -> rht | B norm: exact -> blockscale | C codebook: scalar -> vector |
|---|---|---|---|
| 1 | +0.0008 ± 0.0053 | +0.0008 ± 0.0019 | -0.0174 ± 0.0055 |
| 2 | +0.0016 ± 0.0015 | +0.0006 ± 0.0015 | +0.0194 ± 0.0015 |
| 3 | +0.0020 ± 0.0044 | -0.0015 ± 0.0046 | +0.0233 ± 0.0024 |
| 4 | +0.0001 ± 0.0039 | +0.0009 ± 0.0019 | +0.0133 ± 0.0043 |
| 6 | -0.0009 ± 0.0012 | +0.0005 ± 0.0004 | +0.0062 ± 0.0011 |
| 8 | +0.0011 ± 0.0012 | +0.0014 ± 0.0011 | +0.0014 ± 0.0012 |

**glove100 — cosine**

| bits | A rotation: haar -> rht | B norm: exact -> blockscale | C codebook: scalar -> vector |
|---|---|---|---|
| 1 | +0.0001 ± 0.0021 | +0.0029 ± 0.0003 | +0.0222 ± 0.0021 |
| 2 | +0.0015 ± 0.0013 | +0.0024 ± 0.0010 | +0.0348 ± 0.0015 |
| 3 | -0.0007 ± 0.0012 | +0.0023 ± 0.0015 | +0.0285 ± 0.0012 |
| 4 | +0.0001 ± 0.0005 | +0.0019 ± 0.0011 | +0.0151 ± 0.0011 |
| 6 | +0.0004 ± 0.0003 | -0.0008 ± 0.0003 | +0.0022 ± 0.0003 |
| 8 | -0.0002 ± 0.0006 | +0.0000 ± 0.0006 | +0.0004 ± 0.0006 |

**glove100 — inner product**

| bits | A rotation: haar -> rht | B norm: exact -> blockscale | C codebook: scalar -> vector |
|---|---|---|---|
| 1 | -0.0011 ± 0.0026 | +0.0028 ± 0.0008 | +0.0231 ± 0.0026 |
| 2 | -0.0003 ± 0.0014 | +0.0013 ± 0.0017 | +0.0391 ± 0.0018 |
| 3 | +0.0016 ± 0.0014 | +0.0026 ± 0.0015 | +0.0398 ± 0.0016 |
| 4 | +0.0004 ± 0.0021 | +0.0033 ± 0.0022 | +0.0223 ± 0.0024 |
| 6 | +0.0003 ± 0.0013 | +0.0002 ± 0.0015 | +0.0051 ± 0.0008 |
| 8 | +0.0002 ± 0.0005 | +0.0001 ± 0.0004 | +0.0015 ± 0.0005 |

**nfcorpus1024 — cosine**

| bits | A rotation: haar -> rht | B norm: exact -> blockscale | C codebook: scalar -> vector |
|---|---|---|---|
| 1 | +0.0017 ± 0.0008 | +0.0016 ± 0.0008 | +0.0137 ± 0.0011 |
| 2 | +0.0002 ± 0.0013 | +0.0003 ± 0.0012 | +0.0178 ± 0.0012 |
| 3 | +0.0010 ± 0.0020 | +0.0013 ± 0.0012 | +0.0098 ± 0.0020 |
| 4 | -0.0007 ± 0.0011 | -0.0002 ± 0.0008 | +0.0065 ± 0.0011 |
| 6 | +0.0002 ± 0.0008 | +0.0004 ± 0.0008 | +0.0017 ± 0.0005 |
| 8 | +0.0005 ± 0.0005 | +0.0001 ± 0.0008 | +0.0002 ± 0.0007 |

**nfcorpus1024 — inner product**

| bits | A rotation: haar -> rht | B norm: exact -> blockscale | C codebook: scalar -> vector |
|---|---|---|---|
| 1 | -0.0008 ± 0.0027 | +0.0007 ± 0.0018 | -0.0043 ± 0.0026 |
| 2 | +0.0003 ± 0.0018 | +0.0011 ± 0.0029 | +0.0227 ± 0.0028 |
| 3 | +0.0018 ± 0.0014 | +0.0020 ± 0.0010 | +0.0239 ± 0.0015 |
| 4 | -0.0021 ± 0.0019 | +0.0010 ± 0.0019 | +0.0143 ± 0.0017 |
| 6 | -0.0015 ± 0.0005 | +0.0005 ± 0.0008 | +0.0037 ± 0.0009 |
| 8 | -0.0003 ± 0.0002 | +0.0008 ± 0.0005 | +0.0008 ± 0.0004 |

**fmnist784 — cosine**

| bits | A rotation: haar -> rht | B norm: exact -> blockscale | C codebook: scalar -> vector |
|---|---|---|---|
| 1 | -0.0072 ± 0.0007 | +0.0028 ± 0.0011 | -0.0170 ± 0.0010 |
| 2 | +0.0011 ± 0.0006 | +0.0004 ± 0.0007 | +0.0016 ± 0.0009 |
| 3 | -0.0011 ± 0.0008 | -0.0001 ± 0.0010 | +0.0096 ± 0.0012 |
| 4 | -0.0002 ± 0.0014 | -0.0003 ± 0.0014 | +0.0068 ± 0.0004 |
| 6 | +0.0006 ± 0.0008 | +0.0001 ± 0.0004 | +0.0012 ± 0.0009 |
| 8 | -0.0002 ± 0.0004 | -0.0001 ± 0.0005 | +0.0006 ± 0.0004 |

**fmnist784 — inner product**

| bits | A rotation: haar -> rht | B norm: exact -> blockscale | C codebook: scalar -> vector |
|---|---|---|---|
| 1 | -0.0087 ± 0.0093 | +0.0008 ± 0.0014 | -0.0387 ± 0.0093 |
| 2 | +0.0102 ± 0.0060 | +0.0032 ± 0.0086 | +0.0077 ± 0.0073 |
| 3 | +0.0047 ± 0.0045 | +0.0017 ± 0.0054 | +0.0266 ± 0.0045 |
| 4 | -0.0008 ± 0.0025 | -0.0001 ± 0.0018 | +0.0214 ± 0.0027 |
| 6 | +0.0036 ± 0.0027 | -0.0003 ± 0.0018 | +0.0111 ± 0.0027 |
| 8 | +0.0010 ± 0.0011 | -0.0001 ± 0.0010 | +0.0027 ± 0.0006 |

## Axis summary, pooled over corpora and bit widths

| metric | axis | mean delta R@10 | sd | max |delta| | n cells |
|---|---|---|---|---|---|
| cosine | A rotation: haar -> rht | -0.0004 | 0.0018 | 0.0072 | 24 |
| cosine | B norm: exact -> blockscale | +0.0007 | 0.0011 | 0.0029 | 24 |
| cosine | C codebook: scalar -> vector | +0.0082 | 0.0106 | 0.0348 | 24 |
| inner product | A rotation: haar -> rht | +0.0005 | 0.0031 | 0.0102 | 24 |
| inner product | B norm: exact -> blockscale | +0.0010 | 0.0011 | 0.0033 | 24 |
| inner product | C codebook: scalar -> vector | +0.0112 | 0.0168 | 0.0398 | 24 |

## Controls

| corpus / metric | bits | fp32 | naive uniform (no rotation) | LM+QJL (`prod`) | remex |
|---|---|---|---|---|---|
| arxiv768 / cosine | 1 | 1.000 | 0.701 | n/a | 0.682 |
| arxiv768 / cosine | 2 | 1.000 | 0.838 | 0.698 | 0.828 |
| arxiv768 / cosine | 3 | 1.000 | 0.897 | 0.835 | 0.903 |
| arxiv768 / cosine | 4 | 1.000 | 0.941 | 0.907 | 0.942 |
| arxiv768 / cosine | 6 | 1.000 | 0.968 | 0.970 | 0.982 |
| arxiv768 / cosine | 8 | 1.000 | 0.971 | 0.991 | 0.995 |
| arxiv768 / inner product | 1 | 1.000 | 0.712 | n/a | 0.683 |
| arxiv768 / inner product | 2 | 1.000 | 0.781 | 0.697 | 0.768 |
| arxiv768 / inner product | 3 | 1.000 | 0.861 | 0.775 | 0.847 |
| arxiv768 / inner product | 4 | 1.000 | 0.895 | 0.851 | 0.909 |
| arxiv768 / inner product | 6 | 1.000 | 0.927 | 0.948 | 0.968 |
| arxiv768 / inner product | 8 | 1.000 | 0.931 | 0.981 | 0.987 |
| glove100 / cosine | 1 | 1.000 | 0.310 | n/a | 0.315 |
| glove100 / cosine | 2 | 1.000 | 0.594 | 0.373 | 0.598 |
| glove100 / cosine | 3 | 1.000 | 0.757 | 0.633 | 0.774 |
| glove100 / cosine | 4 | 1.000 | 0.866 | 0.793 | 0.876 |
| glove100 / cosine | 6 | 1.000 | 0.954 | 0.938 | 0.965 |
| glove100 / cosine | 8 | 1.000 | 0.983 | 0.982 | 0.990 |
| glove100 / inner product | 1 | 1.000 | 0.325 | n/a | 0.324 |
| glove100 / inner product | 2 | 1.000 | 0.577 | 0.382 | 0.575 |
| glove100 / inner product | 3 | 1.000 | 0.741 | 0.610 | 0.748 |
| glove100 / inner product | 4 | 1.000 | 0.848 | 0.770 | 0.856 |
| glove100 / inner product | 6 | 1.000 | 0.947 | 0.928 | 0.957 |
| glove100 / inner product | 8 | 1.000 | 0.981 | 0.979 | 0.987 |
| nfcorpus1024 / cosine | 1 | 1.000 | 0.659 | n/a | 0.657 |
| nfcorpus1024 / cosine | 2 | 1.000 | 0.818 | 0.671 | 0.814 |
| nfcorpus1024 / cosine | 3 | 1.000 | 0.897 | 0.820 | 0.896 |
| nfcorpus1024 / cosine | 4 | 1.000 | 0.933 | 0.899 | 0.941 |
| nfcorpus1024 / cosine | 6 | 1.000 | 0.967 | 0.969 | 0.980 |
| nfcorpus1024 / cosine | 8 | 1.000 | 0.973 | 0.990 | 0.994 |
| nfcorpus1024 / inner product | 1 | 1.000 | 0.654 | n/a | 0.652 |
| nfcorpus1024 / inner product | 2 | 1.000 | 0.774 | 0.665 | 0.762 |
| nfcorpus1024 / inner product | 3 | 1.000 | 0.859 | 0.770 | 0.847 |
| nfcorpus1024 / inner product | 4 | 1.000 | 0.897 | 0.850 | 0.912 |
| nfcorpus1024 / inner product | 6 | 1.000 | 0.939 | 0.952 | 0.973 |
| nfcorpus1024 / inner product | 8 | 1.000 | 0.943 | 0.985 | 0.992 |
| fmnist784 / cosine | 1 | 1.000 | 0.095 | n/a | 0.671 |
| fmnist784 / cosine | 2 | 1.000 | 0.504 | 0.686 | 0.793 |
| fmnist784 / cosine | 3 | 1.000 | 0.729 | 0.801 | 0.874 |
| fmnist784 / cosine | 4 | 1.000 | 0.862 | 0.879 | 0.927 |
| fmnist784 / cosine | 6 | 1.000 | 0.956 | 0.961 | 0.977 |
| fmnist784 / cosine | 8 | 1.000 | 0.984 | 0.988 | 0.994 |
| fmnist784 / inner product | 1 | 1.000 | 0.316 | n/a | 0.655 |
| fmnist784 / inner product | 2 | 1.000 | 0.356 | 0.669 | 0.691 |
| fmnist784 / inner product | 3 | 1.000 | 0.253 | 0.701 | 0.771 |
| fmnist784 / inner product | 4 | 1.000 | 0.532 | 0.779 | 0.864 |
| fmnist784 / inner product | 6 | 1.000 | 0.881 | 0.919 | 0.950 |
| fmnist784 / inner product | 8 | 1.000 | 0.976 | 0.974 | 0.985 |

## Actual bytes per vector, itemised

Payload is identical across codebooks by construction; the arms differ only in the side channel. The shared column is the index-level cost (rotation + codebook) amortized over the whole index, not per vector.

The **cosine-opt** column is the honest total for a cosine-only index: documents are unit-norm there, so remex's stored fp32 norm is a constant 1.0 and a real deployment would drop it. The block-scale arm has no such saving — its scales stay live because they carry per-block variance, not just the global norm. Every recall number in this writeup is measured with the norm *stored* (the conservative choice against remex); this column is what remex's byte cost would fall to if it were dropped.

| corpus | arm | bits | payload B | side B | total B | cosine-opt B | shared (KiB) | grid dim m |
|---|---|---|---|---|---|---|---|---|
| arxiv768 | haar+exactnorm+scalar | 1 | 96 | 4 | 100 | 96 | 2304 | 1 |
| arxiv768 | rht+blockscale+vector | 1 | 96 | 12 | 108 | 108 | 16 | 8 |
| arxiv768 | haar+exactnorm+scalar | 2 | 192 | 4 | 196 | 192 | 2304 | 1 |
| arxiv768 | rht+blockscale+vector | 2 | 192 | 12 | 204 | 204 | 2056 | 8 |
| arxiv768 | haar+exactnorm+scalar | 3 | 288 | 4 | 292 | 288 | 2304 | 1 |
| arxiv768 | rht+blockscale+vector | 3 | 288 | 12 | 300 | 300 | 72 | 4 |
| arxiv768 | haar+exactnorm+scalar | 4 | 384 | 4 | 388 | 384 | 2304 | 1 |
| arxiv768 | rht+blockscale+vector | 4 | 384 | 12 | 396 | 396 | 1032 | 4 |
| arxiv768 | haar+exactnorm+scalar | 6 | 576 | 4 | 580 | 576 | 2304 | 1 |
| arxiv768 | rht+blockscale+vector | 6 | 576 | 12 | 588 | 588 | 40 | 2 |
| arxiv768 | haar+exactnorm+scalar | 8 | 768 | 4 | 772 | 768 | 2305 | 1 |
| arxiv768 | rht+blockscale+vector | 8 | 768 | 12 | 780 | 780 | 520 | 2 |
| glove100 | haar+exactnorm+scalar | 1 | 12 | 4 | 16 | 12 | 39 | 1 |
| glove100 | rht+blockscale+vector | 1 | 12 | 4 | 16 | 16 | 3 | 5 |
| glove100 | haar+exactnorm+scalar | 2 | 25 | 4 | 29 | 25 | 39 | 1 |
| glove100 | rht+blockscale+vector | 2 | 25 | 4 | 29 | 29 | 22 | 5 |
| glove100 | haar+exactnorm+scalar | 3 | 38 | 4 | 42 | 38 | 39 | 1 |
| glove100 | rht+blockscale+vector | 3 | 38 | 4 | 42 | 42 | 642 | 5 |
| glove100 | haar+exactnorm+scalar | 4 | 50 | 4 | 54 | 50 | 39 | 1 |
| glove100 | rht+blockscale+vector | 4 | 50 | 10 | 60 | 60 | 1026 | 4 |
| glove100 | haar+exactnorm+scalar | 6 | 75 | 4 | 79 | 75 | 39 | 1 |
| glove100 | rht+blockscale+vector | 6 | 75 | 4 | 79 | 79 | 34 | 2 |
| glove100 | haar+exactnorm+scalar | 8 | 100 | 4 | 104 | 100 | 40 | 1 |
| glove100 | rht+blockscale+vector | 8 | 100 | 4 | 104 | 104 | 514 | 2 |
| nfcorpus1024 | haar+exactnorm+scalar | 1 | 128 | 4 | 132 | 128 | 4096 | 1 |
| nfcorpus1024 | rht+blockscale+vector | 1 | 128 | 16 | 144 | 144 | 13 | 8 |
| nfcorpus1024 | haar+exactnorm+scalar | 2 | 256 | 4 | 260 | 256 | 4096 | 1 |
| nfcorpus1024 | rht+blockscale+vector | 2 | 256 | 16 | 272 | 272 | 2053 | 8 |
| nfcorpus1024 | haar+exactnorm+scalar | 3 | 384 | 4 | 388 | 384 | 4096 | 1 |
| nfcorpus1024 | rht+blockscale+vector | 3 | 384 | 16 | 400 | 400 | 69 | 4 |
| nfcorpus1024 | haar+exactnorm+scalar | 4 | 512 | 4 | 516 | 512 | 4096 | 1 |
| nfcorpus1024 | rht+blockscale+vector | 4 | 512 | 16 | 528 | 528 | 1029 | 4 |
| nfcorpus1024 | haar+exactnorm+scalar | 6 | 768 | 4 | 772 | 768 | 4096 | 1 |
| nfcorpus1024 | rht+blockscale+vector | 6 | 768 | 16 | 784 | 784 | 37 | 2 |
| nfcorpus1024 | haar+exactnorm+scalar | 8 | 1024 | 4 | 1028 | 1024 | 4097 | 1 |
| nfcorpus1024 | rht+blockscale+vector | 8 | 1024 | 16 | 1040 | 1040 | 517 | 2 |
| fmnist784 | haar+exactnorm+scalar | 1 | 98 | 4 | 102 | 98 | 2401 | 1 |
| fmnist784 | rht+blockscale+vector | 1 | 98 | 14 | 112 | 112 | 19 | 8 |
| fmnist784 | haar+exactnorm+scalar | 2 | 196 | 4 | 200 | 196 | 2401 | 1 |
| fmnist784 | rht+blockscale+vector | 2 | 196 | 14 | 210 | 210 | 2059 | 8 |
| fmnist784 | haar+exactnorm+scalar | 3 | 294 | 4 | 298 | 294 | 2401 | 1 |
| fmnist784 | rht+blockscale+vector | 3 | 294 | 14 | 308 | 308 | 75 | 4 |
| fmnist784 | haar+exactnorm+scalar | 4 | 392 | 4 | 396 | 392 | 2401 | 1 |
| fmnist784 | rht+blockscale+vector | 4 | 392 | 14 | 406 | 406 | 1035 | 4 |
| fmnist784 | haar+exactnorm+scalar | 6 | 588 | 4 | 592 | 588 | 2401 | 1 |
| fmnist784 | rht+blockscale+vector | 6 | 588 | 14 | 602 | 602 | 43 | 2 |
| fmnist784 | haar+exactnorm+scalar | 8 | 784 | 4 | 788 | 784 | 2402 | 1 |
| fmnist784 | rht+blockscale+vector | 8 | 784 | 14 | 798 | 798 | 523 | 2 |

## Shared bytes: what amortization actually costs at these corpus sizes

Shared = rotation + codebook, divided by the number of documents in that corpus. `true B/vec` is payload + side + shared. This is the column the headline tables leave out.

| corpus | N | bits | arm | B/vec (headline) | shared B/vec | true B/vec | N for <5% |
|---|---|---|---|---|---|---|---|
| arxiv768 | 750 | 1 | haar+exactnorm+scalar | 100 | 3145.7 | 3245.7 | 471,861 |
| arxiv768 | 750 | 1 | rht+blockscale+vector | 108 | 21.2 | 129.2 | 2,940 |
| arxiv768 | 750 | 2 | haar+exactnorm+scalar | 196 | 3145.7 | 3341.7 | 240,747 |
| arxiv768 | 750 | 2 | rht+blockscale+vector | 204 | 2806.4 | 3010.4 | 206,357 |
| arxiv768 | 750 | 3 | haar+exactnorm+scalar | 292 | 3145.8 | 3437.8 | 161,598 |
| arxiv768 | 750 | 3 | rht+blockscale+vector | 300 | 97.6 | 397.6 | 4,882 |
| arxiv768 | 750 | 4 | haar+exactnorm+scalar | 388 | 3145.8 | 3533.8 | 121,617 |
| arxiv768 | 750 | 4 | rht+blockscale+vector | 396 | 1408.3 | 1804.3 | 53,347 |
| arxiv768 | 750 | 6 | haar+exactnorm+scalar | 580 | 3146.1 | 3726.1 | 81,364 |
| arxiv768 | 750 | 6 | rht+blockscale+vector | 588 | 53.9 | 641.9 | 1,376 |
| arxiv768 | 750 | 8 | haar+exactnorm+scalar | 772 | 3147.1 | 3919.1 | 61,149 |
| arxiv768 | 750 | 8 | rht+blockscale+vector | 780 | 709.3 | 1489.3 | 13,641 |
| glove100 | 20,000 | 1 | haar+exactnorm+scalar | 16 | 2.0 | 18.5 | 48,495 |
| glove100 | 20,000 | 1 | rht+blockscale+vector | 16 | 0.1 | 16.6 | 3,200 |
| glove100 | 20,000 | 2 | haar+exactnorm+scalar | 29 | 2.0 | 31.0 | 27,598 |
| glove100 | 20,000 | 2 | rht+blockscale+vector | 29 | 1.1 | 30.1 | 15,504 |
| glove100 | 20,000 | 3 | haar+exactnorm+scalar | 42 | 2.0 | 43.5 | 19,293 |
| glove100 | 20,000 | 3 | rht+blockscale+vector | 42 | 32.9 | 74.4 | 316,800 |
| glove100 | 20,000 | 4 | haar+exactnorm+scalar | 54 | 2.0 | 56.0 | 14,839 |
| glove100 | 20,000 | 4 | rht+blockscale+vector | 60 | 52.5 | 112.5 | 350,192 |
| glove100 | 20,000 | 6 | haar+exactnorm+scalar | 79 | 2.0 | 81.0 | 10,192 |
| glove100 | 20,000 | 6 | rht+blockscale+vector | 79 | 1.7 | 80.7 | 8,803 |
| glove100 | 20,000 | 8 | haar+exactnorm+scalar | 104 | 2.1 | 106.1 | 7,890 |
| glove100 | 20,000 | 8 | rht+blockscale+vector | 104 | 26.3 | 130.3 | 101,210 |
| nfcorpus1024 | 3,633 | 1 | haar+exactnorm+scalar | 132 | 1154.5 | 1286.5 | 635,502 |
| nfcorpus1024 | 3,633 | 1 | rht+blockscale+vector | 144 | 3.7 | 147.7 | 1,849 |
| nfcorpus1024 | 3,633 | 2 | haar+exactnorm+scalar | 260 | 1154.5 | 1414.5 | 322,640 |
| nfcorpus1024 | 3,633 | 2 | rht+blockscale+vector | 272 | 578.7 | 850.7 | 154,579 |
| nfcorpus1024 | 3,633 | 3 | haar+exactnorm+scalar | 388 | 1154.5 | 1542.5 | 216,203 |
| nfcorpus1024 | 3,633 | 3 | rht+blockscale+vector | 400 | 19.4 | 419.4 | 3,533 |
| nfcorpus1024 | 3,633 | 4 | haar+exactnorm+scalar | 516 | 1154.5 | 1670.5 | 162,573 |
| nfcorpus1024 | 3,633 | 4 | rht+blockscale+vector | 528 | 290.0 | 818.0 | 39,913 |
| nfcorpus1024 | 3,633 | 6 | haar+exactnorm+scalar | 772 | 1154.6 | 1926.6 | 108,668 |
| nfcorpus1024 | 3,633 | 6 | rht+blockscale+vector | 784 | 10.4 | 794.4 | 967 |
| nfcorpus1024 | 3,633 | 8 | haar+exactnorm+scalar | 1028 | 1154.8 | 2182.8 | 81,622 |
| nfcorpus1024 | 3,633 | 8 | rht+blockscale+vector | 1040 | 145.7 | 1185.7 | 10,181 |

## Axis A wall-clock — applying the rotation

The ratio is deliberately not called a speedup. The RHT really is O(d log d) against Haar's O(d^2), but this measures numpy, which runs the dense rotation as one BLAS sgemm and the FWHT as a Python loop over strided slices. d=4096 and 8192 are included, well past any corpus here, so the asymptotic crossover is measured rather than assumed.

| d | vectors | Haar (dense) | RHT | haar/rht | RHT rounds | Haar build | RHT build |
|---|---|---|---|---|---|---|---|
| 100 | 4096 | 0.9 ms | 4.3 ms | 0.22x (Haar) | 4 | 1 ms | 0.31 ms |
| 768 | 4096 | 41.5 ms | 48.3 ms | 0.86x (Haar) | 2 | 75 ms | 0.44 ms |
| 1024 | 4096 | 74.3 ms | 79.7 ms | 0.93x (Haar) | 1 | 145 ms | 0.23 ms |
| 2048 | 512 | 37.1 ms | 22.5 ms | 1.65x (RHT) | 1 | 2321 ms | 0.33 ms |
| 4096 | 512 | 152.5 ms | 49.8 ms | 3.06x (RHT) | 1 | 14106 ms | 0.75 ms |
| 8192 | 512 | 611.7 ms | 138.4 ms | 4.42x (RHT) | 1 | 92764 ms | 0.70 ms |

