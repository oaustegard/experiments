// The Voice Tuner's worker: runs engine.js (on top of tinyvoca.js, the library other pages use) off the main thread.
// The page passes its build id; asking for the scripts under it means a browser never pairs a new page with a
// cached old library.
const BUILD = new URL(location.href).searchParams.get("build") || "";
importScripts("tinyvoca.js?build=" + encodeURIComponent(BUILD), "engine.js?build=" + encodeURIComponent(BUILD));
const handle = makeTunerEngine((message, transfer) => postMessage(message, transfer || []));
onmessage = ({ data }) => handle(data);
