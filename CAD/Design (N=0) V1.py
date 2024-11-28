import cadquery as cq
from ocp_vscode import *

# Dimensions (mm)
radius_main = 2.5
height_main = 15
radius_secondary = 1
height_secondary = 5
n_circular = 7
num_rotations = 5  # Number of small cylinders around the z-axis
num_repetitions = 4 # Number of repetitions of small cylinders along Z-axis
angle_rotation = 360 / num_rotations

# Ensure spacing is greater than the secondary cylinder's radius
spacing = max(3, radius_secondary + 0.1)  # Add a small buffer (0.1 mm) to the radius check

# Main cylinder
main_cylinder = cq.Workplane("XY").circle(radius_main).extrude(height_main)

# Secondary cylinder
secondary_cylinder = (
    cq.Workplane("XZ")
    .workplane(offset=0.9 * radius_main)
    .center(0, radius_main)
    .circle(radius_secondary)
    .extrude(height_secondary)
)

def apply_rotations(shape, num_rotations, rotation_angle, axis_point, axis_direction):
    """Rotates a shape around a central axis."""
    shapes = [shape]
    for _ in range(num_rotations - 1):
        shapes.append(shapes[-1].rotate(axis_point, axis_direction, rotation_angle))
    return shapes

def apply_linear_repetition(shape, num_repetitions, spacing):
    """Creates a linear repetition of a shape along the Z-axis."""
    shapes = [shape]
    for i in range(1, num_repetitions):
        new_shape = shape.translate((0, 0, i * spacing))
        shapes.append(new_shape)
    return shapes

# Apply rotations around the Z-axis
secondary_cylinder_rotated = apply_rotations(
    secondary_cylinder, num_rotations, angle_rotation, (0, 0, 0), (0, 0, 1)
)

# Ensure linear repetitions remain on the main cylinder's surface
secondary_cylinder_repeated = []
for cylinder in secondary_cylinder_rotated:
    # Apply linear repetition with the calculated spacing
    secondary_cylinder_repeated.extend(
        apply_linear_repetition(cylinder, num_repetitions, spacing)
    )

# Combine all secondary cylinders with the main cylinder
combined_model = main_cylinder
for cyl in secondary_cylinder_repeated:
    combined_model = combined_model.union(cyl)

# Show the final model
show_object(combined_model, name="Combined Model")
