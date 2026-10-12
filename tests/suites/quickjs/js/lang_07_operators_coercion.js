console.log(1 + '2', '3' - 1, '3' * '4', [] + [], [] + {}, 1 + null, 1 + undefined, true + true, '5' + 5 - 5);
console.log(0 == '', null == undefined, null == 0, NaN == NaN, [1] == 1, '1,2' == [1, 2], ({}) == '[object Object]');
console.log(typeof null, typeof [], typeof function () {}, typeof Symbol(), typeof 1n, typeof undefined, typeof NaN);
console.log(5 / 2 | 0, -5 >> 1, -5 >>> 28, 2 ** 10, 2 ** -1, (-2) ** 2, 7 % -3, -7 % 3, 1 / 0, -1 / 0, 0 / 0, 0 === -0, Object.is(0, -0));
console.log(~~3.7, ~5, 1 << 31, 1 << 32, 2 ** 53 + 1, 0.1 + 0.2, 9007199254740993, 1e21, 1e-7, 123456789012345680000);
