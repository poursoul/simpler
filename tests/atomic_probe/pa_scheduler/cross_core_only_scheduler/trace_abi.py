"""Frozen Atomic/DCCI ABI subset used by the scheduler-only v1/v2 traces.

Copied from the ordinary converter; IDs match common/pa_model.h. Keep these
local so conversion never imports a parent checkout or another experiment.
"""

ATOMIC_SITE_NAMES = {
    0: "startup_increment",
    1: "startup_poll",
    2: "fatal_poll",
    3: "fatal_set",
    5: "fanin_flag_load",
    43: "shared_exec_fatal_load",
    44: "shared_exec_fatal_set",
    45: "shared_exec_cell_state_load",
    48: "shared_exec_claim",
    49: "shared_exec_completion_vend_publish",
    50: "shared_exec_completion_flag_publish",
    51: "shared_exec_done_publish",
    52: "shared_exec_drain_arrive",
    53: "shared_exec_drain_release_publish",
    54: "shared_exec_drain_release_poll",
    55: "shared_exec_drain_arrival_poll",
    56: "shared_exec_dispatch_ticket",
}
ATOMIC_OP_NAMES = {
    0: "load",
    1: "exchange",
    2: "fetch_add",
    3: "fetch_max",
    4: "compare_exchange",
}
ATOMIC_SITE_OP_IDS = {
    0: 2,
    1: 0,
    2: 0,
    3: 1,
    5: 0,
    43: 0,
    44: 4,
    45: 0,
    48: 4,
    49: 1,
    50: 1,
    51: 4,
    52: 2,
    53: 1,
    54: 0,
    55: 0,
    56: 2,
}
ATOMIC_RETURN_READY = 1 << 6
ATOMIC_POLL_BATCH = 1 << 7

DCCI_SITE_NAMES = {
    8: "observer_trace_export",
    9: "startup_config_invalidate",
    12: "shared_exec_payload_invalidate",
    13: "shared_exec_token_descriptor_invalidate",
}
DCCI_SITE_OP_IDS = {8: 1, 9: 0, 12: 0, 13: 0}
