import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import random
from matplotlib.animation import FuncAnimation, FFMpegWriter
#Configuration
grid_size = 20
total_cells = grid_size * grid_size
num_survivors = 10
battery_capacity = 30
max_steps = 200
swap_cooldown = 5
multi_drones = 5
random.seed(42)
np.random.seed(42)
# Setup
survivors = set()
while len(survivors) < num_survivors:
    pos = (np.random.randint(0, grid_size), np.random.randint(0, grid_size))
    survivors.add(pos)
def move_serpentine(step, band_rows, start_row):
    local = step % (band_rows * grid_size)
    r_local, c = divmod(local, grid_size)
    row = start_row + r_local
    col = c if row % 2 == 0 else grid_size - 1 - c
    return (row, col)

#Multi-Drone Simulation
rows_per_drone = grid_size // multi_drones
bands = [(d * rows_per_drone, (d + 1) * rows_per_drone) for d in range(multi_drones)]
drones = [{
    "id": d,
    "pos": [bands[d][0], 0],
    "battery": battery_capacity,
    "coverage": set(),
    "detected": set(),
    "band": bands[d]
} for d in range(multi_drones)]

helper_available = 0
multi_coverage = []
multi_swap_events = 0
multi_detected = set()
multi_95_step = None
multi_survivor_logs = []
multi_swap_logs = []
for step in range(max_steps):
    for drone in drones:
        if drone["battery"] == 0:
            if step >= helper_available:
                drone["battery"] = battery_capacity
                helper_available = step + swap_cooldown
                multi_swap_events += 1
                multi_swap_logs.append((step, tuple(drone["pos"])))
            else:
                continue

        start, end = drone["band"]
        pos = move_serpentine(step, end - start, start)
        drone["pos"] = pos
        drone["battery"] -= 1
        drone["coverage"].add(pos)
        if pos in survivors and pos not in drone["detected"]:
            drone["detected"].add(pos)
            multi_detected.add(pos)
            multi_survivor_logs.append((step, pos))

    total = set().union(*[d["coverage"] for d in drones])
    pct = len(total) / total_cells * 100
    multi_coverage.append(pct)
    if multi_95_step is None and pct >= 95:
        multi_95_step = step

#Single-Drone Simulation 
single_coverage = []
single_detected = set()
single_swap_events = 0
single_95_step = None
battery = battery_capacity
coverage = set()
pos = [0, 0]
for step in range(max_steps):
    if battery == 0:
        battery = battery_capacity
        single_swap_events += 1

    x, y = divmod(step % total_cells, grid_size)
    pos = [x, y]
    coverage.add((x, y))
    battery -= 1
    if (x, y) in survivors:
        single_detected.add((x, y))

    pct = len(coverage) / total_cells * 100
    single_coverage.append(pct)
    if single_95_step is None and pct >= 95:
        single_95_step = step

# Comparison between Multi-Drone and Single-Drone
plt.figure(figsize=(8, 4))
plt.plot(multi_coverage, label="Multi-Drone + Mid-Air Swap", color="blue")
plt.plot(single_coverage, label="Single Drone + Ground Recharge", color="red")
plt.axhline(95, color="gray", linestyle="--", label="95% Coverage Target")
plt.xlabel("Steps")
plt.ylabel("Coverage (%)")
plt.title("Coverage Comparison: Multi vs Single Drone")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("coverage_comparison.png", dpi=200)
#Heatmap
cmap = mcolors.ListedColormap(['white', 'lightblue', 'red'])
bounds = [0, 0.1, 0.9, 1.1]
norm = mcolors.BoundaryNorm(bounds, cmap.N)
# Multi-drone heatmap
heatmap_multi = np.zeros((grid_size, grid_size))
for d in drones:
    for (x, y) in d["coverage"]:
        heatmap_multi[x, y] = max(heatmap_multi[x, y], 0.5)
for (x, y) in survivors:
    heatmap_multi[x, y] = 1.0
fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(heatmap_multi, cmap=cmap, norm=norm)
ax.set_title("Multi-Drone SAR Heatmap")
ax.set_xticks(range(grid_size))
ax.set_yticks(range(grid_size))
ax.grid(True, alpha=0.3)
for d in drones:
    x, y = d["pos"]
    ax.plot(y, x, 'bo', markersize=8)
plt.tight_layout()
plt.savefig("uav_multi_heatmap.png", dpi=200)
# Single-drone heatmap
heatmap_single = np.zeros((grid_size, grid_size))
for (x, y) in coverage:
    heatmap_single[x, y] = max(heatmap_single[x, y], 0.5)
for (x, y) in survivors:
    heatmap_single[x, y] = 1.0
fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(heatmap_single, cmap=cmap, norm=norm)
ax.set_title("Single Drone SAR Heatmap")
ax.set_xticks(range(grid_size))
ax.set_yticks(range(grid_size))
ax.grid(True, alpha=0.3)
ax.plot(pos[1], pos[0], 'bo', markersize=8)
plt.tight_layout()
plt.savefig("uav_single_heatmap.png", dpi=200)
#Metrics
print("\n📊 Performance Metrics")
print(f"Multi-Drone: 95% coverage at step {multi_95_step}, {len(multi_detected)} survivors detected, {multi_swap_events} swaps")
print(f"Single Drone: 95% coverage at step {single_95_step}, {len(single_detected)} survivors detected, {single_swap_events} recharges")
print("Saved: coverage_comparison.png, uav_multi_heatmap.png, uav_single_heatmap.png")
# Multi-Drone Animation 
detected_steps = {}
for step, pos in multi_survivor_logs:
    detected_steps.setdefault(step, []).append(pos)
swap_steps = {step: pos for step, pos in multi_swap_logs}
fig, ax = plt.subplots(figsize=(6,6))
ax.set_xlim(-0.5, grid_size-0.5)
ax.set_ylim(-0.5, grid_size-0.5)
ax.set_xticks(range(grid_size))
ax.set_yticks(range(grid_size))
ax.grid(True, alpha=0.3)
drone_scat = ax.scatter([], [], c='blue', s=80, label="Drones")
title = ax.text(0.5, 1.03, "", transform=ax.transAxes, ha="center")
survivor_flash = ax.scatter([], [], c='red', s=100, marker='*', label="Survivors Detected")
swap_marker = ax.scatter([], [], c='green', s=100, marker='P', label="Battery Swap")
ax.legend(loc="lower left", fontsize=9, framealpha=0.6)

def frame_positions(frame):
    positions = []
    for d, drone in enumerate(drones):
        start, end = drone["band"]
        pos = move_serpentine(frame, end - start, start)
        positions.append(pos[::-1])  # (col,row)
    return positions
def init():
    drone_scat.set_offsets(np.empty((0,2)))
    survivor_flash.set_offsets(np.empty((0,2)))
    swap_marker.set_offsets(np.empty((0,2)))
    title.set_text("UAV Swarm Animation")
    return drone_scat, survivor_flash, swap_marker, title
def update(frame):
    positions = frame_positions(frame)
    drone_scat.set_offsets(np.array(positions))
    title.set_text(f"Step {frame}")
    flashes = detected_steps.get(frame, [])
    if flashes:
        survivor_flash.set_offsets(np.array([(y, x) for (x, y) in flashes]))
    else:
        survivor_flash.set_offsets(np.empty((0, 2)))
    if frame in swap_steps:
        x, y = swap_steps[frame]
        swap_marker.set_offsets(np.array([[y, x]]))
    else:
        swap_marker.set_offsets(np.empty((0, 2)))

    return drone_scat, survivor_flash, swap_marker, title

ani = FuncAnimation(fig, update, frames=max_steps, init_func=init, blit=False, interval=150)
writer = FFMpegWriter(fps=12)
ani.save("uav_comparo_anim.mp4", writer=writer, dpi=200)
print("\n🎬 Animation saved: uav_comparo_anim.mp4")

#Single-Drone Animation  
fig, ax = plt.subplots(figsize=(6,6))
ax.set_xlim(-0.5, grid_size-0.5)
ax.set_ylim(-0.5, grid_size-0.5)
ax.set_xticks(range(grid_size))
ax.set_yticks(range(grid_size))
ax.grid(True, alpha=0.3)
single_scat = ax.scatter([], [], c='blue', s=80, label="Single Drone")
title = ax.text(0.5, 1.03, "", transform=ax.transAxes, ha="center")
survivor_flash_single = ax.scatter([], [], c='red', s=100, marker='*', label="Survivors Detected")
swap_marker_single = ax.scatter([], [], c='green', s=100, marker='P', label="Battery Recharge")
ax.legend(loc="upper right", fontsize=9, framealpha=0.6)

#Survivor logs for single drone sim
detected_steps_single = {}
swap_steps_single = {}
single_detected = set()
coverage = set()
battery = battery_capacity
pos = [0, 0]
for step in range(max_steps):
    if battery == 0:
        battery = battery_capacity
        single_swap_events += 1
        swap_steps_single[step] = tuple(pos)
    x, y = divmod(step % total_cells, grid_size)
    pos = [x, y]
    coverage.add((x, y))
    battery -= 1
    if (x, y) in survivors and (x, y) not in single_detected:
        single_detected.add((x, y))
        detected_steps_single.setdefault(step, []).append((x, y))
    pct = len(coverage) / total_cells * 100
    single_coverage.append(pct)
    if single_95_step is None and pct >= 95:
        single_95_step = step

def frame_positions_single(frame):
    x, y = divmod(frame % total_cells, grid_size)
    return [(y, x)]  # (col,row)
def init_single():
    single_scat.set_offsets(np.empty((0,2)))
    survivor_flash_single.set_offsets(np.empty((0,2)))
    swap_marker_single.set_offsets(np.empty((0,2)))
    title.set_text("Single Drone Animation")
    return single_scat, survivor_flash_single, swap_marker_single, title
def update_single(frame):
    positions = frame_positions_single(frame)
    single_scat.set_offsets(np.array(positions))
    title.set_text(f"Step {frame}")
    flashes = detected_steps_single.get(frame, [])
    if flashes:
        survivor_flash_single.set_offsets(np.array([(y, x) for (x, y) in flashes]))
    else:
        survivor_flash_single.set_offsets(np.empty((0, 2)))
    if frame in swap_steps_single:
        x, y = swap_steps_single[frame]
        swap_marker_single.set_offsets(np.array([[y, x]]))
    else:
        swap_marker_single.set_offsets(np.empty((0, 2)))

    return single_scat, survivor_flash_single, swap_marker_single, title

ani_single = FuncAnimation(fig, update_single, frames=max_steps, init_func=init_single, blit=False, interval=150)
writer = FFMpegWriter(fps=12)
ani_single.save("uav_single_anim.mp4", writer=writer, dpi=200)
print("🎬 Single-drone animation saved: uav_single_anim.mp4")