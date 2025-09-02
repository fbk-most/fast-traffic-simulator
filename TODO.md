## Ideas and todos

We need to manage:

- multi-edge graphs, either directly or with some tricks (e.g., adding fictitious nodes and edges). At the moment, only
  one among multiple edges is kept.
- vehicles with unreachable destinations
- vehicles starting and ending at the same node

Improvements:

- no need to manage the whole vehicle array in all iterations: we can consider windows of active vehicles (e.g., from
  first active to last active)
