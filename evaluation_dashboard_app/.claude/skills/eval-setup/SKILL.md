---
name: eval-setup
description: Configure and verify the connection to the evaluation dashboard server (URL, token, Cloudflare Access, queue, workers). Use when evalctl commands fail with auth/connection errors, when a Cloudflare Access sign-in is needed, or when setting up a new machine to drive evaluations.
---

# Eval server setup & doctor

Two environment variables drive everything:

- `EVAL_DASHBOARD_URL` — e.g. `http://eval-server:8502` (the backend API port, not
  the Streamlit UI port). A public deployment is an `https://…` hostname instead.
- `EVAL_EXPORT_TOKEN` — only if the server demands a token (`doctor` will say)

If the server sits behind **Cloudflare Access**, see the section below — `evalctl`
handles it on its own, but which credential it uses is worth choosing deliberately.

## Verify

```bash
python scripts/evalctl.py doctor
```

Interpreting it:

- `UNREACHABLE` → wrong URL/port, VPN, or the server is down. Distinguish
  connection-refused (wrong port / not running) from timeout (network/VPN).
- `is behind Cloudflare Access` → see below; nothing is wrong with the server.
- `FAIL authorized` → the reason is printed. "requires an export token" → get the
  token value from the server's `EVAL_EXPORT_TOKEN` env (ask the server admin),
  export it locally, re-run doctor.
- `FAIL task queue` → server-side configuration (USE_TASK_QUEUE/DATABASE_URL/
  REDIS_URL); nothing to fix client-side — report it to whoever runs the server.
- `workers alive: 0` → workflows would queue but never run; the server's RQ worker
  container needs a restart.

## Cloudflare Access

The edge intercepts unauthenticated requests before they ever reach the dashboard, so
a missing Access session looks nothing like a normal auth failure: a `302` to
`*.cloudflareaccess.com/cdn-cgi/access/login/…`, or `HTTP 403: error code: 1010`.
`EVAL_EXPORT_TOKEN` does **not** help — that is an app-level header, and the request
never gets far enough for the app to read it.

`evalctl` detects this itself and recovers without being asked. On a challenge it:

1. sends `CF_ACCESS_CLIENT_ID` / `CF_ACCESS_CLIENT_SECRET` if both are set
   (a **service token** — headless, no browser: what CI and agents should use);
2. otherwise reuses a cached `cloudflared` browser session for that hostname;
3. otherwise runs `cloudflared access login <url>`, which opens a browser, and
   retries with the JWT it mints;
4. retries once more with a forced fresh login if a cached session had expired.

`doctor` prints which of those is in play, e.g.
`cloudflare access: browser session (cloudflared), expires in 21.4h (08-05 09:12 JST)`.

### Choosing the credential

**Interactive user, occasional runs** → nothing to configure; the first command opens
a browser. The session then lasts as long as the Access policy allows (typically
24h). To get it over with up front, or after an expiry:

```bash
python scripts/evalctl.py login          # sign in now
python scripts/evalctl.py login --force  # ignore any cached session
```

Note the browser step needs a human. When driving this for a user in a chat session,
tell them to run it themselves (in Claude Code: `! python scripts/evalctl.py login`)
rather than launching a login they cannot see or complete.

**CI, cron, a headless box, or an agent that cannot use a browser** → service token.
Ask whoever administers the Access application for a token scoped to it, then:

```bash
export CF_ACCESS_CLIENT_ID=<id>.access
export CF_ACCESS_CLIENT_SECRET=<secret>
```

(`EVAL_CF_CLIENT_ID` / `EVAL_CF_CLIENT_SECRET` work too, matching the desktop
client's settings.) A service token is sent proactively on every request, so there is
no challenge round trip and no browser, ever.

To forbid the browser fallback outright — right for unattended jobs, where a hung
login is worse than a fast failure — pass `--no-cf-login` or set
`EVAL_CF_AUTO_LOGIN=0`. The command then fails immediately, printing both fixes.

### When Access itself is the problem

- **`cloudflared` is not installed** → the error says so. Install it from
  Cloudflare's downloads page, or use a service token instead.
- **Login completes but requests are still refused** → the account that signed in is
  not on the Access policy for this application. That is an admin change; nothing
  client-side will fix it.
- **`rejected the service token`** → the token belongs to a different Access
  application or was revoked. `evalctl` does not retry these, since retrying cannot
  help.
- **Login never returns** → it times out after 300s; run
  `cloudflared access login <url>` directly to see the browser prompt.

Do not work around Access by disabling TLS verification or by lifting a cookie out of
a browser profile; both break silently later.

## Persist

Once working, persist the variables in the user's shell profile (ask before editing
their dotfiles), or prefix them per command. Never print a token or client secret back
in full — `doctor` and the error messages already truncate them.

The `cloudflared` session is cached by `cloudflared` itself under `~/.cloudflared/`,
so it survives across shells with nothing to persist.
