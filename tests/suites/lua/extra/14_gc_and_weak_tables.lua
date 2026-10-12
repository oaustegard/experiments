local weakk = setmetatable({}, {__mode = "k"}) local weakv = setmetatable({}, {__mode = "v"})
do local key = {} weakk[key] = 1 weakv[1] = {} weakv[2] = "string stays" weakv[3] = 42 end
collectgarbage() collectgarbage() local n = 0 for _ in pairs(weakk) do n = n + 1 end print(n, weakv[1], weakv[2], weakv[3])
local fin = {} do local obj = setmetatable({}, {__gc = function() fin[#fin + 1] = "finalized" end}) end collectgarbage() collectgarbage() print(#fin, fin[1])
print(type(collectgarbage("count")), collectgarbage("step") ~= nil, collectgarbage("isrunning"), collectgarbage("incremental"), collectgarbage("generational"), collectgarbage("incremental"))
local before = collectgarbage("count") local junk = {} for i = 1, 10000 do junk[i] = {i} end local mid = collectgarbage("count") junk = nil collectgarbage() local after = collectgarbage("count") print(mid > before, after < mid)
local cache = setmetatable({}, {__mode = "kv"}) for i = 1, 100 do cache[i] = {i} end collectgarbage() local alive = 0 for _ in pairs(cache) do alive = alive + 1 end print(alive)
local t = setmetatable({}, {__gc = function(o) print("gc at close") end}) t = nil
local big = {} for i = 1, 100 do big[i] = string.rep("x", 1000) .. i end print(#big, #big[100])
local ephem = setmetatable({}, {__mode = "k"}) do local k = {} ephem[k] = {ref = k} end collectgarbage() print(next(ephem))
