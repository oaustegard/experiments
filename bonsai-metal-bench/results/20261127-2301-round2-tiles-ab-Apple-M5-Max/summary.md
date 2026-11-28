# Task 2 (rebuild extension), interleaved: stock against each variant

Fork 43f31f400c, PQ2_0, tensor API on, `-p 2048,8192 -r 3`, 45 s cooldown between runs. Each variant is compared with the mean of the stock runs on either side of it.

| run | variant | pp2048 tok/s | pp8192 tok/s |
|---|---|---|---|
| 0 | stock | 753.005935 | 609.605744 |
| 1 | nk4 | 662.860126 | 640.14447 |
| 2 | stock | 703.472687 | 658.809264 |
| 3 | nk8 | 724.414516 | 678.746341 |
| 4 | stock | 650.670804 | 625.84749 |
| 5 | nk4 | 652.419056 | 614.406505 |
| 6 | stock | 530.359889 | 576.540497 |
| 7 | nk8 | 548.283042 | 582.047087 |
| 8 | stock | 628.984141 | 599.018228 |

| variant | pp2048 vs adjacent stock | pp8192 vs adjacent stock |
|---|---|---|
| nk4 | -9.0%, +10.5% | +0.9%, +2.2% |
| nk8 | +7.0%, -5.4% | +5.7%, -1.0% |
