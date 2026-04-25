"""
Global sensor-channel configuration.

Supports per-case sensor selection while keeping one shared source of truth.
"""

# Per-case sensor nodes.
# case1: keep legacy setting (N1, N7)
# case2: use all seven points
CASE_SENSOR_NODES = {
    "case1": [1, 7],
    "case2": [1, 2, 3, 4, 5, 6, 7],
}


def build_disp_cols(sensor_nodes):
    nodes = sensor_nodes
    return [f"N{n}_UZ" for n in nodes]


def build_acc_cols(sensor_nodes):
    nodes = sensor_nodes
    return [f"N{n}_AZ" for n in nodes]


def get_sensor_nodes(case_name: str):
    if case_name not in CASE_SENSOR_NODES:
        raise KeyError(f"Unknown case name: {case_name!r}")
    return CASE_SENSOR_NODES[case_name]
