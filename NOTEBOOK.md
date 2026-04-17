# Notebook

## Open questions / things to revisit

- `transactions.json` items contain a nested `documents` list (contract note, cost info URLs). Currently kept as-is (serialized) in the CSV. Consider whether to drop, expand into separate columns, or leave.
- `download_document()` is implemented but needs `SC_TOKEN`. The CLI uses `https://de.scalable.capital/api/download?id=...` (GraphQL at `/api/cli/graphql`). Token source unknown — not in keychain, not in config files. Try intercepting with mitmproxy once cert pinning is bypassed, or inspect the `sc` binary.
