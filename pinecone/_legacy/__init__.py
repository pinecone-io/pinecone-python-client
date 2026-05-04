"""Internal package holding backcompat shims that adapt legacy execution
models (e.g. ``async_req=True`` thread-pool dispatch) onto the canonical
methods. Opt-in only — never imported on the fast path.

:meta private:
"""
