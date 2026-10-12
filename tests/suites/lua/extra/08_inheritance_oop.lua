local Animal = {} Animal.__index = Animal
function Animal.new(name, sound) return setmetatable({name = name, sound = sound}, Animal) end
function Animal:speak() return self.name .. " says " .. self.sound end
local Dog = setmetatable({}, {__index = Animal}) Dog.__index = Dog
function Dog.new(name) local d = Animal.new(name, "woof") return setmetatable(d, Dog) end
function Dog:fetch() return self.name .. " fetches" end
local d = Dog.new("Rex") print(d:speak(), d:fetch(), getmetatable(d) == Dog, rawget(Dog, "speak"), d.nothing)
local Stack = {} Stack.__index = Stack function Stack.new() return setmetatable({n = 0}, Stack) end
function Stack:push(v) self.n = self.n + 1 self[self.n] = v return self end function Stack:pop() local v = self[self.n] self[self.n] = nil self.n = self.n - 1 return v end
local s = Stack.new():push(1):push(2):push(3) print(s:pop(), s:pop(), s.n)
local Counter = setmetatable({}, {__call = function(cls, start) return setmetatable({v = start}, cls) end}) Counter.__index = Counter
function Counter:next() self.v = self.v + 1 return self.v end print(Counter(10):next(), Counter(0):next())
