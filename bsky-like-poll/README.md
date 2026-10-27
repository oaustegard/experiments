# Like Poll

A poll builder for Bluesky, which has no polls. You sign in with a Bluesky
handle and app password, type a question and 2–10 options, and the page posts
a thread: the question with a one-line instruction on how to vote, then one
reply per option. People vote by liking the option they want.

Live: <https://austegard.com/experiments/bsky-like-poll/>

## How the thread is built

- The question post ends with `🗳️ Poll: open the thread and ♥ like ONE reply
  to vote.` and `🧵 N options below 👇`, so someone who sees it in a feed knows
  to open the thread.
- Each option is `1️⃣ Tacos` plus `♥ like to vote`, posted as a chain (each
  option replies to the one before it). Bluesky shows an author's own reply
  chain in order under the root, which sibling replies don't guarantee.
- By default a threadgate with an empty allow list turns off replies from
  everyone else, so the thread holds only the options. Quote posts still work.
  The gate is written after the options, so it can't block the author's own
  replies.
- The preview counts graphemes against Bluesky's 300 limit and disables
  posting when any post is over.

If a post fails partway, the page says how many went up and links the root so
you can delete them.

## Counting

Likes are deduplicated per post, not across options, so one account can like
every option. The Results box takes the question post's link, walks the
author's numbered reply chain, pages through `app.bsky.feed.getLikes` for each
option, and shows two columns: **Likes** (raw) and **Votes** (likes from
accounts that liked exactly one option). It reads the public AppView, so it
needs no sign-in, and `?poll=<post link>` loads it directly.

## Limits

- Sign-in uses an app password through `com.atproto.server.createSession`
  against the user's own PDS (resolved from the handle's DID document). There
  is no server: the password goes only to the PDS, and the session tokens sit
  in `sessionStorage` until the tab closes. OAuth would avoid handling a
  password at all but needs a hosted client-metadata document and redirect
  flow; app passwords were the smallest working version.
- Links, mentions and hashtags in the question are posted as plain text (no
  facets).
- Votes are public: anyone can see who liked which option.
- Counting depends on the option posts keeping their keycap prefix
  (`1️⃣`…`🔟`); editing isn't possible on Bluesky, so that holds.

## Testing

`node test_poll.js` drives the page in headless Chromium with the network
mocked. It signs in, builds a three-option poll, and checks the five records
written (root, three chained replies with correct `root`/`parent` refs, and the
threadgate keyed to the root's rkey). It then counts a mocked thread in which
one account liked two options and one option's likes span two pages, and
checks both columns. It passes. The real AppView endpoints (`getPostThread`,
`getLikes`) were checked by hand for response shape. Nothing was posted to a
live account, so the first real poll is also the first end-to-end run.
