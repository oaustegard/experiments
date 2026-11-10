# Cloudflare's MCP server reaches 3,763 API operations through four tools

Cloudflare's MCP server exposes four tools: `whoami`, `docs`, `search` and `execute`. Two of them, `search` and `execute`, take a JavaScript function as input instead of named parameters, and together they reach the whole Cloudflare REST API. The numbers and behaviour below come from probing the server from a Claude Code session in November 2026.

## whoami

Takes no arguments and returns the authenticated user's id and email. The tool description says the id stays the same across token refresh, reconnection and permission changes, so it works as a check on which identity a write will run as.

## docs

Semantic search over the Cloudflare product documentation (Workers, R2, D1, Durable Objects, Zero Trust, DNS, CDN and the rest). It takes a plain query string and returns matching chunks. It is the only one of the four that does not take code.

## search

`search` runs an async arrow function against a `spec` object, which holds Cloudflare's OpenAPI document with every `$ref` already resolved inline. A count over it returned:

- 2,345 paths and 3,763 operations
- 724 distinct tags; the largest are `cloudforce-one` (294 operations), `access` (185), `magic` (144), `workers` (138) and `dlp` (108)

Each operation carries an `x-api-token-group` field listing the token permissions it accepts, so the permission an endpoint needs can be read before calling it.

The function can filter and reshape the spec however it likes, and only its return value enters the model's context:

```js
async () => {
  const out = [];
  for (const [path, methods] of Object.entries(spec.paths))
    for (const [method, op] of Object.entries(methods))
      if (op.tags?.includes('r2') && method === 'patch')
        out.push({ path, summary: op.summary, body: op.requestBody });
  return out;
}
```

## execute

`execute` runs an async arrow function with two globals:

- `cloudflare.request({ method, path, query, body, contentType })` calls the API as the authenticated user. `request` serialises plain objects and arrays as JSON; `FormData`, `Blob`, `Uint8Array` and streams are sent as they are. Binary responses come back as `{ base64, byteLength }`.
- `accountId` is filled in when the session is authorised for exactly one account. With several, pass `account_id` to the tool, or call `GET /accounts` to find them.

What the sandbox allows, from probing it:

- **Several API calls in one tool call.** A script can list zones, loop over them, update records and return a summary, with no model turn between the steps.
- **Multipart uploads.** The tool's own example deploys a Worker with a KV binding in one `PUT` using `FormData`.
- **Outbound `fetch`.** A request to `https://example.com` succeeded, so a script can pull data from the web and write it into KV, R2 or D1 in the same run. The same path lets a script send account data to any host.
- **No `require` and no `process`.** There are no npm modules and no environment variables. `setTimeout` is available.
- **Errors that name the missing permission.** A 401 or 403 throws an error that names the method, path, status, the permission the endpoint needs and whether reconnecting would help.
- **A result cap of about 6,000 tokens**, which `truncate: false` lifts. The description recommends lifting it only when the output goes to a file.

## Design

This is the "code mode" pattern. A tool per endpoint would put thousands of schemas into the model's context before the first call. Here the spec costs nothing until a `search` function reads part of it, and the model's ability to write JavaScript replaces a large tool list.

`execute` has no read-only mode. A `DELETE` has the same call shape as a `GET`, and the tool schema gives a client no way to tell which one a given script will send. Any limit on writes has to come from the client's permission prompt for the tool or from the scopes granted to the OAuth token. I have not tested which scopes a default connection grants.
