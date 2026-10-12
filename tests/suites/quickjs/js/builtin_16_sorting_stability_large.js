const n = 2000; const a = Array.from({ length: n }, (_, i) => ({ k: (i * 7919) % 101, i })); a.sort((x, y) => x.k - y.k);
let ok = true; for (let i = 1; i < n; i++) if (a[i - 1].k === a[i].k && a[i - 1].i > a[i].i) ok = false; console.log('stable', ok, a[0], a[n - 1]);
const nums = Array.from({ length: 1000 }, (_, i) => (i * 2654435761) % 1000003); nums.sort((x, y) => x - y); console.log(nums.slice(0, 5), nums[999], nums.every((v, i) => i == 0 || nums[i - 1] <= v));
const strs = nums.slice(0, 50).map(String).sort(); console.log(strs.slice(0, 5), [5, 25, 100, 1].sort(), [5, 25, 100, 1].sort((a, b) => a - b), ['b', undefined, 'a', , 'c'].sort(), [3, 2, 1].sort(() => 0));
