import numpy as np

from .crafting_utils import (
    get_basic_inventory_counts,
    ensure_have_items,
    inv_gui_coords,
    slot_center,
    table_gui_coords,
    grid_xy,
    hotbar_xy,
    tick,
    look,
)

# Hotbar layout 
HOTBAR_LOGS   = 1   
HOTBAR_PLANKS = 2  
HOTBAR_STICKS = 3   
HOTBAR_TABLE  = 4   
HOTBAR_AXEOUT = 5   

# Next slot to place a crafted axe
NEXT_AXE_SLOT = HOTBAR_AXEOUT

# Axe crafting cap: at most 5 axes in hotbar slots 5–9
MAX_AXES_IN_HOTBAR = 5
AXES_CRAFTED_THIS_EPISODE = 0

TABLE_GUI_OPEN = False


def _get_tracer_obs(env):
    """ Retrieves the most recent observation stored in the environment wrapper """
    if hasattr(env, "last") and env.last is not None:
        obs, _, _, _ = env.last
        return obs
    return None


def _install_reset_hook(env):
    """Every episode reset also resets NEXT_AXE_SLOT and axe counters."""
    current = env
    while current is not None:
        if not getattr(current, "_axe_reset_hook_installed", False):
            original_reset = current.reset

            def make_wrapped(orig):
                def wrapped_reset(*args, **kwargs):
                    global NEXT_AXE_SLOT, AXES_CRAFTED_THIS_EPISODE, TABLE_GUI_OPEN
                    NEXT_AXE_SLOT = HOTBAR_AXEOUT
                    AXES_CRAFTED_THIS_EPISODE = 0
                    TABLE_GUI_OPEN = False
                    return orig(*args, **kwargs)
                return wrapped_reset

            current.reset = make_wrapped(original_reset)
            current._axe_reset_hook_installed = True

        current = getattr(current, "env", None)

def craft_planks_from_logs(env, helper, logs_to_convert=3, width=640, height=360, obs=None):
    """ Inventory 2×2: Logs -> Planks. Checks requirements if it can craft """
    global TABLE_GUI_OPEN

    # If the 3x3 crafting table GUI is open, close it first.
    if TABLE_GUI_OPEN:
        helper.toggle_inventory(); tick(env, 2)
        TABLE_GUI_OPEN = False
        look(env, pitch=-7.0, repeats=6)

    if obs is not None:
        required_logs = max(3, logs_to_convert)
        if not ensure_have_items({"logs": required_logs}, obs):
            return False

    tl, craft2_tl, craft2_out, inv_tl, hotbar_tl = inv_gui_coords(width, height)

    helper.toggle_inventory(); tick(env, 2)

    lx, ly = hotbar_xy(tl, hotbar_tl, HOTBAR_LOGS - 1)
    helper.move_to(lx, ly); helper.left_click(); tick(env, 1)

    gx, gy = grid_xy(tl, craft2_tl, 0, 0)
    for _ in range(logs_to_convert):
        helper.move_to(gx, gy); helper.right_click(); tick(env, 1)

    helper.move_to(lx, ly); helper.left_click(); tick(env, 1)

    planks_slot = HOTBAR_PLANKS
    ox, oy = slot_center(tl, craft2_out[0], craft2_out[1])
    px, py = hotbar_xy(tl, hotbar_tl, planks_slot - 1)
    for _ in range(logs_to_convert):
        helper.move_to(ox, oy); helper.left_click(); tick(env, 1)
        helper.move_to(px, py); helper.left_click(); tick(env, 1)

    helper.toggle_inventory(); tick(env, 2)

    return True

def craft_sticks_in_inventory(env, helper, width=640, height=360, obs=None):
    """ Inventory 2x2: Sticks from Planks. Checks requirements if it can craft. """
    global TABLE_GUI_OPEN

    # If the 3x3 crafting table GUI is open, close it first.
    if TABLE_GUI_OPEN:
        helper.toggle_inventory(); tick(env, 2)
        TABLE_GUI_OPEN = False
        look(env, pitch=-7.0, repeats=6)

    if obs is not None:
        if not ensure_have_items({"planks": 2}, obs):
            return False

    tl, craft2_tl, craft2_out, inv_tl, hotbar_tl = inv_gui_coords(width, height)

    helper.toggle_inventory(); tick(env, 2)

    planks_slot = HOTBAR_PLANKS
    px, py = hotbar_xy(tl, hotbar_tl, planks_slot - 1)
    helper.move_to(px, py); helper.left_click(); tick(env, 1)

    for r, c in [(0, 0), (1, 0)]:
        gx, gy = grid_xy(tl, craft2_tl, r, c)
        helper.move_to(gx, gy); helper.right_click(); tick(env, 1)

    helper.move_to(px, py); helper.left_click(); tick(env, 1)

    sticks_slot = HOTBAR_STICKS
    ox, oy = slot_center(tl, craft2_out[0], craft2_out[1])
    helper.move_to(ox, oy); helper.left_click(); tick(env, 1)
    sx, sy = hotbar_xy(tl, hotbar_tl, sticks_slot - 1)
    helper.move_to(sx, sy); helper.left_click(); tick(env, 2)

    helper.toggle_inventory(); tick(env, 2)

    return True


def craft_table_in_inventory(env, helper, width=640, height=360, obs=None):
    """ Inventory 2×2: Make Crafting Table from Planks. Checks requirements if it can craft """
    if obs is not None:
        if not ensure_have_items({"planks": 4}, obs):
            return False

    tl, craft2_tl, craft2_out, inv_tl, hotbar_tl = inv_gui_coords(width, height)

    helper.toggle_inventory(); tick(env, 2)

    planks_slot = HOTBAR_PLANKS
    px, py = hotbar_xy(tl, hotbar_tl, planks_slot - 1)
    helper.move_to(px, py); helper.left_click(); tick(env, 1)

    for r, c in [(0,0), (0,1), (1,0), (1,1)]:
        gx, gy = grid_xy(tl, craft2_tl, r, c)
        helper.move_to(gx, gy); helper.right_click(); tick(env, 1)

    helper.move_to(px, py); helper.left_click(); tick(env, 1)

    table_slot = HOTBAR_TABLE
    ox, oy = slot_center(tl, craft2_out[0], craft2_out[1])
    helper.move_to(ox, oy); helper.left_click(); tick(env, 1)
    hx, hy = hotbar_xy(tl, hotbar_tl, table_slot - 1)
    helper.move_to(hx, hy); helper.left_click(); tick(env, 2)

    helper.toggle_inventory(); tick(env, 2)

    table_slot = HOTBAR_TABLE
    helper.select_hotbar(table_slot)

    look(env, pitch=7.0, repeats=6)

    helper.left_click();  tick(env, 1)
    helper.left_click();  tick(env, 1)
    helper.right_click(); tick(env, 4)
    helper.right_click(); tick(env, 3)

    global TABLE_GUI_OPEN
    TABLE_GUI_OPEN = True

    return True

def craft_wooden_axe(env, helper, width=640, height=360, obs=None):
    """ Table 3×3: Wooden Axe from Planks + Sticks. Checks requirements if it can craft """
    global NEXT_AXE_SLOT, AXES_CRAFTED_THIS_EPISODE, TABLE_GUI_OPEN

    if not TABLE_GUI_OPEN:
        print("[crafting_guide] Not in crafting table 3x3 GUI. Aborting craft_wooden_axe.")
        return False
    
    if obs is not None:
        if AXES_CRAFTED_THIS_EPISODE >= MAX_AXES_IN_HOTBAR:
            print("[crafting_guide] Axe cap reached (>= 5 axes in hotbar). Aborting craft_wooden_axe.")
            helper.toggle_inventory()
            TABLE_GUI_OPEN = False
            look(env, pitch=-7.0, repeats=6)
            return False
        if not ensure_have_items({"planks": 3, "sticks": 2}, obs):
            helper.toggle_inventory()
            TABLE_GUI_OPEN = False
            look(env, pitch=-7.0, repeats=6)
            return False

    tl, craft3_tl, craft3_out, inv_tl, hotbar_tl = table_gui_coords(width, height)

    planks_slot = HOTBAR_PLANKS
    px, py = hotbar_xy(tl, hotbar_tl, planks_slot - 1)
    helper.move_to(px, py); helper.left_click(); tick(env, 1)
    for r, c in [(0,0), (0,1), (1,0)]:
        gx, gy = grid_xy(tl, craft3_tl, r, c)
        helper.move_to(gx, gy); helper.right_click(); tick(env, 1)
    helper.move_to(px, py); helper.left_click(); tick(env, 1)

    sticks_slot = HOTBAR_STICKS
    sx, sy = hotbar_xy(tl, hotbar_tl, sticks_slot - 1)
    helper.move_to(sx, sy); helper.left_click(); tick(env, 1)
    for r, c in [(1,1), (2,1)]:
        gx, gy = grid_xy(tl, craft3_tl, r, c)
        helper.move_to(gx, gy); helper.right_click(); tick(env, 1)
    helper.move_to(sx, sy); helper.left_click(); tick(env, 1)

    axe_slot = NEXT_AXE_SLOT
    ox, oy = slot_center(tl, craft3_out[0], craft3_out[1])
    helper.move_to(ox, oy); helper.left_click(); tick(env, 1)
    ax, ay = hotbar_xy(tl, hotbar_tl, axe_slot - 1)
    helper.move_to(ax, ay); helper.left_click(); tick(env, 2)

    helper.toggle_inventory()
    TABLE_GUI_OPEN = False
    helper.select_hotbar(axe_slot)
    look(env, pitch=-7.0, repeats=6)

    AXES_CRAFTED_THIS_EPISODE += 1
    if NEXT_AXE_SLOT < 9:
        NEXT_AXE_SLOT += 1

    return True


def craft_pipeline_make_and_equip_axe(env, helper, logs_to_convert=3, width=640, height=360, obs=None):
    """ Runs the full sequence to craft to wooden axe. Checks requirements if it can craft """
    global AXES_CRAFTED_THIS_EPISODE

    _install_reset_hook(env)

    if obs is not None:
        if AXES_CRAFTED_THIS_EPISODE >= MAX_AXES_IN_HOTBAR:
            print("[crafting_guide] Axe cap reached (>= 5 axes in hotbar). Aborting pipeline.")
            return False

        required_logs = max(3, logs_to_convert)
        if not ensure_have_items({"logs": required_logs}, obs):
            print("[crafting_guide] Not enough logs to start full axe pipeline.")
            return False

    current_obs = obs

    ok = craft_planks_from_logs(
        env, helper,
        logs_to_convert=logs_to_convert,
        width=width, height=height,
        obs=current_obs,
    )
    if not ok:
        return False

    current_obs = _get_tracer_obs(env)

    ok = craft_sticks_in_inventory(
        env, helper,
        width=width, height=height,
        obs=current_obs,
    )

    current_obs = _get_tracer_obs(env)

    ok = craft_table_in_inventory(
        env, helper,
        width=width, height=height,
        obs=current_obs,
    )
    if not ok:
        return False

    current_obs = _get_tracer_obs(env)

    ok = craft_wooden_axe(
        env, helper,
        width=width, height=height,
        obs=current_obs,
    )
    if not ok:
        return False

    return True
