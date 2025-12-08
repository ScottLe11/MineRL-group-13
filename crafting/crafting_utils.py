import numpy as np

# GUI Geometry
GUI_W, GUI_H = 176, 166
SLOT = 18

# All logs variants
LOG_ITEM_KEYS = [
    "oak_log",
    "birch_log",
    "spruce_log",
    "jungle_log",
    "acacia_log",
    "dark_oak_log",
    "oak_wood",
    "birch_wood",
    "spruce_wood",
    "jungle_wood",
    "acacia_wood",
    "dark_oak_wood",
]

# All plank variants
PLANK_ITEM_KEYS = [
    "oak_planks",
    "birch_planks",
    "spruce_planks",
    "jungle_planks",
    "acacia_planks",
    "dark_oak_planks",
    "crimson_planks",
    "warped_planks",
]

STICK_ITEM_KEYS = ["stick"]
TABLE_ITEM_KEYS = ["crafting_table"]
AXE_ITEM_KEYS   = ["wooden_axe"]

LOGICAL_TO_KEYS = {
    "logs":           LOG_ITEM_KEYS,
    "planks":         PLANK_ITEM_KEYS,
    "sticks":         STICK_ITEM_KEYS,
    "crafting_table": TABLE_ITEM_KEYS,
    "wooden_axe":     AXE_ITEM_KEYS,
}


def _inventory_from_obs(obs):
    """ Extract a plain dict[str, int] from obs['inventory']. """
    if not isinstance(obs, dict) or "inventory" not in obs:
        return {}
    inv_raw = obs["inventory"]
    inv = {}
    for k, v in inv_raw.items():
        try:
            inv[k] = int(v)
        except Exception:
            inv[k] = int(np.array(v).reshape(()))
    return inv


def _count_any(inv, keys):
    """Sum counts for any of the given item keys."""
    return int(sum(inv.get(k, 0) for k in keys))


def get_basic_inventory_counts(obs):
    """ Returns a small summary dict of key items, aggregating all wood variants """
    inv = _inventory_from_obs(obs)
    return {
        "logs":           _count_any(inv, LOG_ITEM_KEYS),
        "planks":         _count_any(inv, PLANK_ITEM_KEYS),
        "sticks":         _count_any(inv, STICK_ITEM_KEYS),
        "crafting_table": _count_any(inv, TABLE_ITEM_KEYS),
        "wooden_axe":     _count_any(inv, AXE_ITEM_KEYS),
    }


def debug_inventory(obs, prefix="[crafting_guide]"):
    """ Convenience helper to print a compact view of the inventory """
    if obs is None:
        #print(f"{prefix} No observation provided.")
        return
    counts = get_basic_inventory_counts(obs)
    # print(
    #     f"{prefix} logs={counts['logs']}  planks={counts['planks']}  "
    #     f"sticks={counts['sticks']}  table={counts['crafting_table']}  "
    #     f"wooden_axes={counts['wooden_axe']}"
    # )


def ensure_have_items(required, obs, prefix="[crafting_guide]"):
    """ Check we have enough items to craft something. """
    if obs is None:
        return True

    inv = _inventory_from_obs(obs)
    ok = True
    for logical_name, needed in required.items():
        keys = LOGICAL_TO_KEYS.get(logical_name, [logical_name])
        have = _count_any(inv, keys)
        if have < needed:
            # print(
            #     f"{prefix} Not enough {logical_name}: "
            #     f"need {needed}, have {have}. Aborting craft."
            # )
            ok = False

    return ok

# GUI helpers
def gui_top_left(W, H):
    """Returns the top-left pixel coordinates of the centered GUI window."""
    return (W // 2 - GUI_W // 2, H // 2 - GUI_H // 2)


def slot_center(tl, x, y):
    """Calculates the center pixel coordinates of a specific slot based on offset."""
    return (tl[0] + x + 8, tl[1] + y + 8)


def inv_gui_coords(width=640, height=360):
    """Returns the top-left coordinates for the main GUI, crafting grid, output, inventory, and hotbar."""
    tl = gui_top_left(width, height)
    craft2_tl  = (98, 16)
    craft2_out = (154, 28)
    inv_tl     = (8, 84)
    hotbar_tl  = (8, 142)
    return tl, craft2_tl, craft2_out, inv_tl, hotbar_tl


def table_gui_coords(width=640, height=360):
    """Returns the coordinate set specifically for the 3x3 crafting table interface."""
    tl = gui_top_left(width, height)
    craft3_tl  = (30, 17)
    craft3_out = (124, 35)
    inv_tl     = (8, 84)
    hotbar_tl  = (8, 142)
    return tl, craft3_tl, craft3_out, inv_tl, hotbar_tl


def grid_xy(tl, grid_tl, r, c):
    """Returns the center pixel of a slot at row r and column c within a specific crafting grid."""
    return slot_center(tl, grid_tl[0] + c * SLOT, grid_tl[1] + r * SLOT)


def inv_xy(tl, inv_tl, r, c):
    """Returns the center pixel of a slot at row r and column c within the main player inventory."""
    return slot_center(tl, inv_tl[0] + c * SLOT, inv_tl[1] + r * SLOT)


def hotbar_xy(tl, hotbar_tl, col):
    """Returns the center pixel of the specified column index in the hotbar."""
    return slot_center(tl, hotbar_tl[0] + col * SLOT, hotbar_tl[1])


# Frame helpers
def tick(env, steps=2, render=True):
    """Advance the env with no-op for a few frames to allow GUI to update."""
    for _ in range(steps):
        if render:
            env.render()
        env.step(env.action_space.no_op())


def look(env, pitch=0.0, yaw=0.0, repeats=1, render=True):
    """Send small camera deltas (pitch,yaw) for `repeats` frames."""
    for _ in range(repeats):
        act = env.action_space.no_op()
        act["camera"] = np.array([pitch, yaw], dtype=np.float32)
        if render:
            env.render()
        env.step(act)
