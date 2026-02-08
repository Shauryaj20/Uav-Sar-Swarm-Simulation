import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import random
import time
from matplotlib.animation import FuncAnimation, FFMpegWriter

# Configuration 
grid_size = 20              # Grid N x N
num_drones = 6              # Fleet size
num_survivors = 14          # Survivors randomly placed
battery_capacity = 30       # Energy units per drone
max_steps = 180             # Simulation length
swap_cooldown_steps = 5     # Helper drone needs cooldown between swaps

#Initialization 
random.seed(42)
np.random.seed(42)

# Place survivors randomly
survivor_positions = set()
while len(survivor_positions) < num_survivors:
    pos = (np.random.randint(0, grid_size), np.random.randint(0, grid_size))
    survivor_positions.add(pos)

# Assign horizontal bands per drone to reduce overlap
rows_per_drone = max(1, grid_size // num_drones)
bands = []
for d in range(num_drones):
    start_row = d * rows_per_drone
    end_row = grid_size if d == num_drones - 1 else start_row + rows_per_drone
    bands.append((start_row, end_row))
# Initialize drones
drones = [{
    "id": d,
    "pos": [bands[d][0], 0],
    "battery": battery_capacity,
    "coverage": set(),
    "detected": set(),
    "band": bands[d]
} for d in range(num_drones)]
# Helper drone (mid-air battery swap)
helper_available_at_step = 0
def swap_battery(drone, current_step):
    global helper_available_at_step
    if current_step >= helper_available_at_step:
        drone["battery"] = battery_capacity
        helper_available_at_step = current_step + swap_cooldown_steps
        return True
    return False
# Serpentine movement within assigned band
def move_drone(drone, step):
    start_row, end_row = drone["band"]
    band_rows = end_row - start_row
    band_cells = band_rows * grid_size
    local = step % max(band_cells, 1)
    r_local, c = divmod(local, grid_size)
    row = start_row + r_local
    col = c if row % 2 == 0 else (grid_size - 1 - c)
    drone["pos"] = [row, col]

#Simulation loop 
start_time = time.time()
coverage_over_time = []
survivor_logs = []   # store (step, position)
swap_logs = []       # store (step, position)

for step in range(max_steps):
    for drone in drones:
        if drone["battery"] == 0:
            swapped = swap_battery(drone, step)
            if swapped:
                swap_logs.append((step, tuple(drone["pos"])))
            else:
                continue

        move_drone(drone, step)
        x, y = drone["pos"]
        drone["battery"] -= 1
        drone["coverage"].add((x, y))
        if (x, y) in survivor_positions and (x, y) not in drone["detected"]:
            drone["detected"].add((x, y))
            survivor_logs.append((step, (x, y)))
    total_coverage = set().union(*[d["coverage"] for d in drones])
    coverage_pct = len(total_coverage) / (grid_size * grid_size) * 100
    coverage_over_time.append(coverage_pct)

print(f"\nSimulation runtime: {time.time() - start_time:.2f} seconds")

# Final stats
total_coverage = set().union(*[d["coverage"] for d in drones])
total_detected = set().union(*[d["detected"] for d in drones])
final_coverage = len(total_coverage) / (grid_size * grid_size) * 100

# Heatmap
heatmap = np.zeros((grid_size, grid_size))
for (x, y) in total_coverage:
    heatmap[x, y] = max(heatmap[x, y], 0.5)
for (x, y) in survivor_positions:
    heatmap[x, y] = 1.0

fig1, ax1 = plt.subplots(figsize=(6, 6))
cmap = mcolors.ListedColormap(['white', 'lightblue', 'red'])
bounds = [0, 0.1, 0.9, 1.1]
norm = mcolors.BoundaryNorm(bounds, cmap.N)
ax1.imshow(heatmap, cmap=cmap, norm=norm)
ax1.set_title(f"UAV SAR Heatmap ({grid_size}x{grid_size}) ~~ (coverage-blue), (survivors-red), (drones-blue dots)")
ax1.set_xticks(range(grid_size)); ax1.set_yticks(range(grid_size))
ax1.grid(True, color='gray', alpha=0.3)
for drone in drones:
    x, y = drone["pos"]
    ax1.plot(y, x, 'bo', markersize=8)
plt.tight_layout()
plt.savefig("uav_sar_heatmap.png", dpi=150)

#Uav_Coverage chart
fig2, ax2 = plt.subplots(figsize=(8, 4))
ax2.plot(range(len(coverage_over_time)), coverage_over_time, label="Coverage %", color="#1f77b4")
ax2.set_title(f"Coverage Over Time ({num_drones} drones, {num_survivors} survivors)")
ax2.set_xlabel("Step"); ax2.set_ylabel("Coverage (%)")
ax2.set_ylim(0, 100); ax2.grid(True, alpha=0.3); ax2.legend()
plt.tight_layout()
plt.savefig("uav_coverage_chart.png", dpi=150)
plt.show()

# Console summary
print("Survivor Detection Log:")
for step, pos in survivor_logs:
    print(f"Step {step}: Survivor detected at {pos}")

print("\nSwap Events:")
for step, pos in swap_logs:
    print(f"Step {step}: Helper swapped battery at {pos}")

print(f"\nFinal Coverage: {final_coverage:.2f}%")
print(f"Survivors Detected: {len(total_detected)} / {num_survivors}")
print("Saved: uav_sar_heatmap.png, uav_coverage_chart.png")

# Create lookup sets for animation
detected_steps = {}
for step, pos in survivor_logs:
    detected_steps.setdefault(step, []).append(pos)

swap_steps = {step: pos for step, pos in swap_logs}

#Animation of Drone Movement 
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(-0.5, grid_size - 0.5)
ax.set_ylim(-0.5, grid_size - 0.5)
ax.set_xticks(range(grid_size)); ax.set_yticks(range(grid_size))
ax.grid(True, alpha=0.3)
drone_scat = ax.scatter([], [], c='blue', s=80, label="Drones")
title = ax.text(0.5, 1.03, "", transform=ax.transAxes, ha="center")
survivor_flash = ax.scatter([], [], c='red', s=100, marker='*', label="Survivors Detected")
swap_marker = ax.scatter([], [], c='green', s=100, marker='P', label="Battery Swap")
ax.legend(loc="upper right", fontsize=8, framealpha=0.6)
# Reset drone positions for animation
for d, drone in enumerate(drones):
    drone["pos"] = [bands[d][0], 0]
def frame_positions(frame):
    positions = []
    for drone in drones:
        move_drone(drone, frame)
        x, y = drone["pos"]
        positions.append((y, x))  # (col, row)
    return positions
def init():
    drone_scat.set_offsets(np.empty((0, 2)))
    survivor_flash.set_offsets(np.empty((0, 2)))
    swap_marker.set_offsets(np.empty((0, 2)))
    title.set_text("UAV Swarm Animation")
    return drone_scat, survivor_flash, swap_marker, title
def update(frame):
    positions = frame_positions(frame)
    drone_scat.set_offsets(np.array(positions))
    title.set_text(f"Step {frame}")
    # Survivor flashes
    flashes = detected_steps.get(frame, [])
    if flashes:
        survivor_flash.set_offsets(np.array([(y, x) for (x, y) in flashes]))
    else:
        survivor_flash.set_offsets(np.empty((0, 2)))
    # Swap indicator
    if frame in swap_steps:
        x, y = swap_steps[frame]
        swap_marker.set_offsets(np.array([[y, x]]))
    else:
        swap_marker.set_offsets(np.empty((0, 2)))

    return drone_scat, survivor_flash, swap_marker, title

ani = FuncAnimation(fig, update, frames=max_steps, init_func=init, blit=False, interval=150)
writer = FFMpegWriter(fps=12)
ani.save("uav_simulation_anim.mp4", writer=writer,dpi=200)
print("Saved: uav_simulation_anim.mp4")