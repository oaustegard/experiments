local mt = {}
mt.__add = function(a, b) return "add" end mt.__sub = function() return "sub" end mt.__mul = function() return "mul" end mt.__div = function() return "div" end
mt.__mod = function() return "mod" end mt.__pow = function() return "pow" end mt.__unm = function() return "unm" end mt.__idiv = function() return "idiv" end
mt.__band = function() return "band" end mt.__bor = function() return "bor" end mt.__bxor = function() return "bxor" end mt.__shl = function() return "shl" end mt.__shr = function() return "shr" end mt.__bnot = function() return "bnot" end
mt.__concat = function() return "concat" end mt.__len = function() return 42 end mt.__eq = function() return true end mt.__lt = function() return true end mt.__le = function() return false end
mt.__call = function(self, ...) return "call", ... end mt.__tostring = function() return "OBJ" end mt.__name = "MyType"
local a, b = setmetatable({}, mt), setmetatable({}, mt)
print(a + 1, 1 - a, a * a, a / 2, a % 2, a ^ 2, -a, a // 2, a & 1, 1 | a, a ~ 1, a << 1, a >> 1, ~a)
print(a .. "x", "x" .. a, #a, a == b, a ~= b, a < b, a <= b, a > b, a >= b, a(1, 2), tostring(a))
print(rawequal(a, b), a == 1, pcall(function() return a < 1 end))
local log = {} local p = setmetatable({}, {__index = function(t, k) log[#log + 1] = "get " .. k return k .. "!" end, __newindex = function(t, k, v) log[#log + 1] = "set " .. k rawset(t, k, v) end})
print(p.x, p.y) p.z = 1 p.z = 2 print(table.concat(log, ","), p.z)
local ro = setmetatable({}, {__newindex = function() error("readonly", 2) end}) print(pcall(function() ro.x = 1 end))
print(getmetatable("abc").__index == string, getmetatable(setmetatable({}, {__metatable = "locked"})), pcall(setmetatable, setmetatable({}, {__metatable = 1}), {}))
