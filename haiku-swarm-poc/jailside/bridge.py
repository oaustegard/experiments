"""Inside the jail: forward TCP 127.0.0.1:PORT to the gateway's Unix socket.

The jail's network namespace holds only `lo`, so OpenEvolve cannot reach any
host. A path-based Unix socket is a filesystem object, not a network one, so
it crosses the namespace boundary; this is the only way out, and the gateway
behind it only answers chat completions.
"""
import asyncio
import sys

PORT, SOCK = int(sys.argv[1]), sys.argv[2]


async def pipe(r, w):
    try:
        while data := await r.read(65536):
            w.write(data)
            await w.drain()
    except (ConnectionError, asyncio.CancelledError):
        pass
    finally:
        w.close()


async def handle(cr, cw):
    ur, uw = await asyncio.open_unix_connection(SOCK)
    await asyncio.gather(pipe(cr, uw), pipe(ur, cw))


async def main():
    srv = await asyncio.start_server(handle, "127.0.0.1", PORT)
    async with srv:
        await srv.serve_forever()

asyncio.run(main())
