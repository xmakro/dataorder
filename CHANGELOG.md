# Changelog

Orders are part of the API: a release that changes the elements any configuration yields,
or the serialized form of a configuration, is a breaking change (a new minor version while
the crate is 0.x) and says so here. Golden tests in `tests/golden.rs` pin the orders.

## 0.1.0 (unreleased)

First release: `Seq` (sources, concat, mix with sampling schedules, weighted mix, shuffle,
repeat, skip, take, stride), `Order` with random access and seekable cursors, the `serde`
feature.
