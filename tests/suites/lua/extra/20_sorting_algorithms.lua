local function qsort(t, lo, hi) lo, hi = lo or 1, hi or #t if lo < hi then local p, i = t[hi], lo - 1 for j = lo, hi - 1 do if t[j] <= p then i = i + 1 t[i], t[j] = t[j], t[i] end end t[i + 1], t[hi] = t[hi], t[i + 1] qsort(t, lo, i) qsort(t, i + 2, hi) end return t end
local data = {} local x = 12345 for i = 1, 300 do x = (x * 1103515245 + 12345) % 2147483648 data[i] = x % 1000 end
local a, b = {table.unpack(data)}, {table.unpack(data)} qsort(a) table.sort(b) local same = true for i = 1, #a do same = same and a[i] == b[i] end print(same, a[1], a[300], a[150])
local function msort(t) if #t <= 1 then return t end local m = #t // 2 local l, r = msort({table.unpack(t, 1, m)}), msort({table.unpack(t, m + 1)}) local o, i, j = {}, 1, 1 while i <= #l and j <= #r do if l[i] <= r[j] then o[#o + 1] = l[i] i = i + 1 else o[#o + 1] = r[j] j = j + 1 end end for k = i, #l do o[#o + 1] = l[k] end for k = j, #r do o[#o + 1] = r[k] end return o end
local c = msort(data) local ok = true for i = 1, #c do ok = ok and c[i] == b[i] end print(ok)
local words = {"banana", "apple", "Cherry", "date", "Elder", "fig"} table.sort(words, function(p, q) return p:lower() < q:lower() end) print(table.concat(words, " "))
local recs = {} for i = 1, 20 do recs[i] = {k = i % 3, i = i} end table.sort(recs, function(p, q) if p.k ~= q.k then return p.k < q.k end return p.i < q.i end) local out = {} for _, r in ipairs(recs) do out[#out + 1] = r.k .. ":" .. r.i end print(table.concat(out, " "))
print(pcall(table.sort, {3, 1, 2}, function(p, q) return true end))
