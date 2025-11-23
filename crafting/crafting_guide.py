import numpy as np
import cv2 

# Hotbar layout 
HOTBAR_LOGS   = 1   # assumed location of logs 

# fallback if couldnt map spot
HOTBAR_PLANKS = 2   
HOTBAR_STICKS = 3   
HOTBAR_TABLE  = 4  
HOTBAR_AXEOUT = 5  

# GUI Geometry
GUI_W, GUI_H = 176, 166
SLOT = 18

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

# Helper Functions
def tick(env, steps=2, render=True):
    """Advance the env with no-op for a few frames to allow GUI to update."""
    for _ in range(steps):
        if render: env.render()
        env.step(env.action_space.no_op())

def look(env, pitch=0.0, yaw=0.0, repeats=1, render=True):
    """Send small camera deltas (pitch,yaw) for `repeats` frames."""
    for _ in range(repeats):
        act = env.action_space.no_op()
        act["camera"] = np.array([pitch, yaw], dtype=np.float32)
        if render: env.render()
        env.step(act)

# Hotbar scanning & mapping
def _safe_patch(frame, cx, cy, half=6):
    """Return a small (2*half+1)^2 RGB patch around (cx,cy), clamped to frame bounds."""
    h, w = frame.shape[:2]
    x0, x1 = max(0, cx-half), min(w-1, cx+half)
    y0, y1 = max(0, cy-half), min(h-1, cy+half)
    return frame[y0:y1+1, x0:x1+1]

def _scan_empty_hotbar_slots(env, width=640, height=360, in_table=False, std_thresh=24.0):
    """ Returns a list of empty, 1-based hotbar slots by checking pixel variance in the currently open GUI. """
    frame_getter = getattr(env, "get_last_full_frame", None)
    frame = frame_getter() if callable(frame_getter) else None
    if frame is None:
        try:
            frame = env.unwrapped.get_last_full_frame()
        except Exception:
            return [] 

    if in_table:
        tl, _c3_tl, _c3_out, _inv_tl, hotbar_tl = table_gui_coords(width, height)
    else:
        tl, _c2_tl, _c2_out, _inv_tl, hotbar_tl = inv_gui_coords(width, height)

    empty_slots = []
    for col in range(9): 
        cx, cy = hotbar_xy(tl, hotbar_tl, col)
        patch = _safe_patch(frame, cx, cy, half=6)  
        if patch.size == 0:
            continue
        if np.std(patch.astype(np.float32)) < std_thresh:
            empty_slots.append(col + 1)
    return empty_slots

def _choose_free_hotbar_slot(env, hotbar_map, width=640, height=360, in_table=False):
    """Finds the first available hotbar slot, falling back to slot 9 if all are occupied."""
    reserved = set(hotbar_map.values())
    empties = _scan_empty_hotbar_slots(env, width, height, in_table=in_table)
    for s in empties:
        if s not in reserved:
            return s
    return 9  

def _assign_if_missing(hotbar_map, key, slot):
    """Record item if slot mapping is not already present."""
    if key not in hotbar_map:
        hotbar_map[key] = slot

# Inventory 2×2: Logs -> Planks
def craft_planks_from_logs(env, helper, logs_to_convert=3, width=640, height=360, hotbar_map=None):
    """Converts logs from the hotbar into planks using the player's 2x2 inventory grid."""
    if hotbar_map is None: hotbar_map = {}

    tl, craft2_tl, craft2_out, inv_tl, hotbar_tl = inv_gui_coords(width, height)

    helper.toggle_inventory(); tick(env, 2)

    lx, ly = hotbar_xy(tl, hotbar_tl, HOTBAR_LOGS - 1)
    helper.move_to(lx, ly); helper.left_click(); tick(env, 1)

    gx, gy = grid_xy(tl, craft2_tl, 0, 0)
    for _ in range(logs_to_convert):
        helper.move_to(gx, gy); helper.right_click(); tick(env, 1)

    helper.move_to(lx, ly); helper.left_click(); tick(env, 1)

    target_slot = hotbar_map.get("planks")
    if target_slot is None:
        target_slot = _choose_free_hotbar_slot(env, hotbar_map, width, height, in_table=False)
        _assign_if_missing(hotbar_map, "planks", target_slot)

    ox, oy = slot_center(tl, craft2_out[0], craft2_out[1])
    px, py = hotbar_xy(tl, hotbar_tl, target_slot - 1)
    for _ in range(logs_to_convert):
        helper.move_to(ox, oy); helper.left_click(); tick(env, 1)
        helper.move_to(px, py); helper.left_click(); tick(env, 1)

# Inventory 2×2: Make Crafting Table from Planks
def craft_table_in_inventory(env, helper, width=640, height=360, hotbar_map=None):
    """Crafts a crafting table using four planks in the 2x2 grid and moves it to the hotbar."""
    if hotbar_map is None: hotbar_map = {}

    tl, craft2_tl, craft2_out, inv_tl, hotbar_tl = inv_gui_coords(width, height)

    planks_slot = hotbar_map.get("planks", HOTBAR_PLANKS)
    px, py = hotbar_xy(tl, hotbar_tl, planks_slot - 1)
    helper.move_to(px, py); helper.left_click(); tick(env, 1)

    for r, c in [(0,0), (0,1), (1,0), (1,1)]:
        gx, gy = grid_xy(tl, craft2_tl, r, c)
        helper.move_to(gx, gy); helper.right_click(); tick(env, 1)

    helper.move_to(px, py); helper.left_click(); tick(env, 1)

    table_slot = hotbar_map.get("table")
    if table_slot is None:
        table_slot = _choose_free_hotbar_slot(env, hotbar_map, width, height, in_table=False)
        _assign_if_missing(hotbar_map, "table", table_slot)

    ox, oy = slot_center(tl, craft2_out[0], craft2_out[1])
    helper.move_to(ox, oy); helper.left_click(); tick(env, 1)
    hx, hy = hotbar_xy(tl, hotbar_tl, table_slot - 1)
    helper.move_to(hx, hy); helper.left_click(); tick(env, 2)

    helper.toggle_inventory(); tick(env, 2)

# Place + open the crafting table
def place_and_open_table(env, helper, hotbar_map=None):
    """Selects the crafting table, places it on the ground, and opens its interface."""
    if hotbar_map is None: hotbar_map = {}

    table_slot = hotbar_map.get("table", HOTBAR_TABLE) 
    helper.select_hotbar(table_slot)

    look(env, pitch=7.0, repeats=6)  

    helper.left_click(); tick(env, 1)  
    helper.left_click(); tick(env, 1)  
    helper.right_click(); tick(env, 4) 
    helper.right_click(); tick(env, 3)

# Table 3×3: Sticks from Planks
def craft_sticks_in_table(env, helper, width=640, height=360, hotbar_map=None):
    """Crafts sticks by two vertical planks within the 3x3 table, then puts them into a free hotbar slot."""
    if hotbar_map is None: hotbar_map = {}

    tl, craft3_tl, craft3_out, inv_tl, hotbar_tl = table_gui_coords(width, height)

    planks_slot = hotbar_map.get("planks", HOTBAR_PLANKS)
    px, py = hotbar_xy(tl, hotbar_tl, planks_slot - 1)
    helper.move_to(px, py); helper.left_click(); tick(env, 1)

    for r, c in [(0,1), (1,1)]:
        gx, gy = grid_xy(tl, craft3_tl, r, c)
        helper.move_to(gx, gy); helper.right_click(); tick(env, 1)

    helper.move_to(px, py); helper.left_click(); tick(env, 1)

    sticks_slot = hotbar_map.get("sticks")
    if sticks_slot is None:
        sticks_slot = _choose_free_hotbar_slot(env, hotbar_map, width, height, in_table=True)
        _assign_if_missing(hotbar_map, "sticks", sticks_slot)

    ox, oy = slot_center(tl, craft3_out[0], craft3_out[1])
    helper.move_to(ox, oy); helper.left_click(); tick(env, 1)
    sx, sy = hotbar_xy(tl, hotbar_tl, sticks_slot - 1)
    helper.move_to(sx, sy); helper.left_click(); tick(env, 2)

# Table 3×3: Wooden Axe from Planks + Sticks
def craft_wooden_axe(env, helper, width=640, height=360, hotbar_map=None):
    """Crafts a wooden axe using the recipe in the 3x3 table and equips it."""
    if hotbar_map is None: hotbar_map = {}

    tl, craft3_tl, craft3_out, inv_tl, hotbar_tl = table_gui_coords(width, height)

    planks_slot = hotbar_map.get("planks", HOTBAR_PLANKS)
    px, py = hotbar_xy(tl, hotbar_tl, planks_slot - 1)
    helper.move_to(px, py); helper.left_click(); tick(env, 1)
    for r, c in [(0,0), (0,1), (1,0)]:
        gx, gy = grid_xy(tl, craft3_tl, r, c)
        helper.move_to(gx, gy); helper.right_click(); tick(env, 1)
    helper.move_to(px, py); helper.left_click(); tick(env, 1)

    sticks_slot = hotbar_map.get("sticks", HOTBAR_STICKS)
    sx, sy = hotbar_xy(tl, hotbar_tl, sticks_slot - 1)
    helper.move_to(sx, sy); helper.left_click(); tick(env, 1)
    for r, c in [(1,1), (2,1)]:
        gx, gy = grid_xy(tl, craft3_tl, r, c)
        helper.move_to(gx, gy); helper.right_click(); tick(env, 1)
    helper.move_to(sx, sy); helper.left_click(); tick(env, 1)

    axe_slot = hotbar_map.get("axe")
    if axe_slot is None:
        axe_slot = _choose_free_hotbar_slot(env, hotbar_map, width, height, in_table=True)
        _assign_if_missing(hotbar_map, "axe", axe_slot)

    ox, oy = slot_center(tl, craft3_out[0], craft3_out[1])
    helper.move_to(ox, oy); helper.left_click(); tick(env, 1)
    ax, ay = hotbar_xy(tl, hotbar_tl, axe_slot - 1)
    helper.move_to(ax, ay); helper.left_click(); tick(env, 2)

    helper.toggle_inventory()
    helper.select_hotbar(axe_slot)
    look(env, pitch=-7.0, repeats=6)

# Pipeline
def craft_pipeline_make_and_equip_axe(env, helper, logs_to_convert=3, width=640, height=360, hotbar_map=None):
    """Runs the full sequence to craft planks, a table, sticks, and an axe, then equips the axe."""
    if hotbar_map is None:
        hotbar_map = {}

    craft_planks_from_logs(env, helper, logs_to_convert=logs_to_convert, width=width, height=height, hotbar_map=hotbar_map)
    craft_table_in_inventory(env, helper, width=width, height=height, hotbar_map=hotbar_map)
    place_and_open_table(env, helper, hotbar_map=hotbar_map)
    craft_sticks_in_table(env, helper, width=width, height=height, hotbar_map=hotbar_map)
    craft_wooden_axe(env, helper, width=width, height=height, hotbar_map=hotbar_map)