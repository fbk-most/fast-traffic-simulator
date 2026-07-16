# Ideas and todos

We need to manage:

- multi-edge graphs, either directly or with some tricks (e.g., adding fictitious nodes and edges). At the moment, only
  one among multiple edges is kept.
- vehicles with unreachable destinations (at the moment, origin is used as destination for these vehicles).
- vehicles starting and ending at the same node.

Improvements:

- event-driven stepping core (deferred): positions of unobstructed vehicles are closed-form, so per-step work could
  scale with state *changes* rather than in-flight vehicles; modest payoff now that stepping is a minor share of the
  runtime.
- expose per-step started/entered/arrived vehicle *index lists* (in addition to the boolean masks) so large-scale trip
  logging does not need full-array `nonzero()` scans each step.

Done:

- ~~no need to manage the whole vehicle array in all iterations: we can consider windows of active vehicles~~
  (active-vehicle index set, 2026-07)
- ~~automated shortest-path refresh, possibly parallelized~~ (`refresh_interval`/`refresh_workers` build parameters,
  2026-07; note: scipy's Dijkstra holds the GIL, so parallelism uses worker processes, not threads)
