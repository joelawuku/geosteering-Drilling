import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import Akima1DInterpolator
from scipy.interpolate import interp1d as scipyinterp1d
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
from matplotlib.pyplot import imread
from matplotlib.patches import Patch

# Load the rig icon image
rig_icon = imread('rig_icon.png')

# Rotate the image 180 degrees using numpy
rotated_rig_icon = np.rot90(rig_icon, 2)

# Increase the size of the image (e.g., double its dimensions)
scaled_rig_icon = np.repeat(np.repeat(rotated_rig_icon, 2, axis=0), 3, axis=1)

# Prompt the user for the number of coordinates
num_coords = int(input("Enter the number of coordinates: "))

# Initialize an empty array to store the coordinates
target_coords = []

# Loop to input the coordinates
for i in range(num_coords):
    x, y, z = map(float, input(f"Enter coordinates for point {i + 1} (x y z): ").split())
    target_coords.append([x, y, z])

# Convert the list of coordinates to a NumPy array
target_coords = np.array(target_coords)

# Prompt the user for the tvd_kop
tvd_kop = float(input("Enter the tvd_kop: "))

# Sort the coordinates based on the third column (z-axis)
sorted_target_indices = np.argsort(target_coords[:, 2])
target_coords = target_coords[sorted_target_indices]

# Initialize surface_coords as all zeros
surface_coords = np.array([0, 0, 0])

# Surface to KOP coordinates
surface_x, surface_y, surface_z = surface_coords
kop_x, kop_y, kop_z = surface_x, surface_y, tvd_kop

targets_x = target_coords[:, 0]
targets_y = target_coords[:, 1]
targets_z = target_coords[:, 2]

interpolating_x = np.insert(targets_x, 0, kop_x)
interpolating_y = np.insert(targets_y, 0, kop_y)
interpolating_z = np.insert(targets_z, 0, kop_z)


def _akima1DInterp(x_coords, y_coords, z_coords):
    interp_func_x = Akima1DInterpolator(z_coords, x_coords)
    interp_func_y = Akima1DInterpolator(z_coords, y_coords)
    interp_func_z = Akima1DInterpolator(z_coords, z_coords)

    # Define the z values along the curve
    z_interp = np.arange(z_coords[0], z_coords[-1], 10)

    # Calculate the x, y, and z coordinates along the curve
    x_interp = interp_func_x(z_interp)
    y_interp = interp_func_y(z_interp)
    z_interp = interp_func_z(z_interp)

    return x_interp, y_interp, z_interp


def calAzimuthInc(x_coords, y_coords, z_coords) -> tuple():
    delta_x = np.diff(x_coords)
    delta_y = np.diff(y_coords)
    delta_z = np.diff(z_coords)
    horizontal_distance = np.sqrt(delta_x ** 2 + delta_y ** 2)
    vertical_distance = delta_z

    incrad = np.arctan2(horizontal_distance, vertical_distance)
    incrad = np.concatenate((np.array([0]), incrad))
    incdeg = np.rad2deg(incrad)

    azrad = np.arctan2(delta_y, delta_x)
    azrad = np.concatenate((np.array([0]), (azrad)))
    azdeg = np.rad2deg(azrad)

    return ((incrad, incdeg), (azrad, azdeg))


def calculateMeasuredDepth(x_coords, y_coords, z_coords):
    delta_x = np.diff(x_coords)
    delta_y = np.diff(y_coords)
    delta_z = np.diff(z_coords)
    segment_lengths = np.sqrt(delta_x ** 2 + delta_y ** 2 + delta_z ** 2)
    measured_depth = np.cumsum(segment_lengths)
    measured_depth = np.concatenate(([0], measured_depth))

    return measured_depth


x_coords, y_coords, z_coords = _akima1DInterp(interpolating_x, interpolating_y, interpolating_z)

surface_to_kop_zs = np.arange(surface_z, kop_z, 10)
z_coords = np.concatenate((surface_to_kop_zs, z_coords))
x_coords = np.concatenate((np.full((len(surface_to_kop_zs),), surface_x), x_coords))
y_coords = np.concatenate((np.full((len(surface_to_kop_zs),), surface_y), y_coords))

inclination, azimuth = calAzimuthInc(x_coords, y_coords, z_coords)
measured_depth = calculateMeasuredDepth(x_coords, y_coords, z_coords)

header = "EAST     NORTH     TVD        INC      AZI        MD"
print(header)
for idx, z in enumerate(z_coords):
    print(
        f"{x_coords[idx]:.2f}    {y_coords[idx]:.2f}    {z:.2f}    {inclination[0][idx]:.2f}    {azimuth[0][idx]:.2f}    {measured_depth[idx]:.2f}")

# ===========================================================================================================

# Update the number of formations to 100
num_formations = 100

# Generate distinct colors for each formation
depths = np.linspace(0, 10000, num_formations)
colors = np.random.choice(['yellow', 'gray', 'green', 'blue', 'brown'], num_formations)

# Create a new figure and subplot for the animated 2D plot
fig_animated = plt.figure()
ax_animated = fig_animated.add_subplot(111)

# Set labels for the axes
ax_animated.set_xlabel('Horizontal Displacement (ft)')
ax_animated.set_ylabel('Depth (ft)')

# Invert the y-axis for better visualization
ax_animated.invert_yaxis()

ax_animated.set_ylim(np.max(z_coords) + 20, -100)
ax_animated.set_xlim(-20, np.max(x_coords) + 20)

# Set the title
plt.title('')


# Plot the targets
target_markers = ax_animated.scatter(target_coords[:, 0], target_coords[:, 2], c='r', marker='o', label='Targets')

# Plot the colored rectangles for formations
for i in range(num_formations):
    formation_depth = depths[i]
    color = colors[i]

    rect = Rectangle((-20, formation_depth), np.max(x_coords) + 40, np.max(z_coords) + 100, color=color, alpha=0.5,
                     label=f'Formation {i + 1}')
    ax_animated.add_patch(rect)

# Add markers for specified points with annotations
points_to_mark = [(0, 0, 100, 'Steer right'),
                  (140, 152, 1500, 'Hold'),
                  (241, 250, 2000, 'Steer right'),
                  (315, 325, 3000, 'Hold'),
                  (500, 452, 3500, 'Hold')]

for point in points_to_mark:
    x, y, z, label = point
    ax_animated.scatter(x, z, c='red', marker='*', s=100, label=label)
    ax_animated.annotate(label, (x, z), textcoords="offset points", xytext=(0, 10), ha='center')

# Initialize the line for the simulated well path
line_deviated, = ax_animated.plot([], [], c='gray', lw=5, solid_capstyle='round', label='Geo-steered Path')

# Initialize the rough, larger line for drilling
rough_line_deviated, = ax_animated.plot([], [], c='black', lw=6, alpha=0.5, solid_capstyle='round',
                                        label='Drilling Hole')

# Display the rig icon at y=0
rig_x = 0
rig_z = -50
ax_animated.imshow(scaled_rig_icon, extent=[rig_x - 20, rig_x + 20, rig_z - 50, rig_z + 50], aspect='auto')


# Function to update the animated plot
def update(frame):
    x_data = x_coords[:frame]
    z_data = z_coords[:frame]

    line_deviated.set_data(x_data, z_data)

    # Simulate a rough drilling hole by adding noise to the path
    noise_scale = 1
    x_rough = x_data + np.random.normal(0, noise_scale, frame)
    z_rough = z_data + np.random.normal(0, noise_scale, frame)
    rough_line_deviated.set_data(x_rough, z_rough)

    return line_deviated, rough_line_deviated


# Number of frames (adjust this based on your data)
num_frames = len(x_coords)

# Create the animation
ani = FuncAnimation(fig_animated, update, frames=num_frames, interval=100, blit=True, repeat=False)

# Define the legend entries
formation_legend = [
    Patch(facecolor='yellow', edgecolor='black', label='Sandstone'),
    Patch(facecolor='blue', edgecolor='black', label='Carbonate'),
    Patch(facecolor='brown', edgecolor='black', label='Limestone'),
    Patch(facecolor='red', edgecolor='black', label='Dolomite'),
    Patch(facecolor='green', edgecolor='black', label='Shale'),
    Patch(facecolor='gray', edgecolor='black', label='Chalk'),
]

# Add the custom legend to the plot
ax_animated.legend(handles=formation_legend, title="Formations", loc='upper right')

# Show the animation
plt.tight_layout()
plt.show()
