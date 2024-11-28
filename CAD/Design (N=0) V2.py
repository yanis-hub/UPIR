import cadquery as cq
from cadquery import exporters
from ocp_vscode import *

# Dimensions (mm)
radius_main = 2.5
height_main = 15
radius_secondary = 1
height_secondary = 5
num_rotations = 5  # Number of small cylinders around the z-axis
num_repetitions = 5  # Number of repetitions of small cylinders along Z-axis
angle_rotation = 360 / num_rotations
tol = 0.1  # Tolerance for the spacing condition

# Fonction pour calculer ou valider le spacing
def validate_or_compute_spacing(user_spacing=None):
    """
    Valide un spacing personnalisé ou calcule un spacing uniforme.

    Args:
        user_spacing (float, optional): Spacing personnalisé fourni par l'utilisateur.

    Returns:
        float: Spacing valide.
    """
    if user_spacing is not None:
        # Valider le spacing personnalisé
        if user_spacing < (radius_secondary + tol):
            raise ValueError(f"Spacing {user_spacing} is too small. It must be greater than {radius_secondary + tol} mm.")
        if height_main < (height_main / (num_repetitions + 1) + user_spacing * (num_repetitions - 1) + radius_secondary):
            raise ValueError(f"Spacing {user_spacing} is too big. Small cylinders exceed the height : {height_main} mm.")
        return user_spacing
    
    else:
        # Calculer le spacing uniforme
        return height_main / (num_repetitions + 1)

# Choix de l'utilisateur : définir un spacing personnalisé ou garder un spacing uniforme 
use_uniform_spacing = True  # Modifier à False si un spacing personnalisé est souhaité
user_spacing = 3.5  # Exemple de spacing personnalisé (modifiable si use_uniform_spacing = False)

# Calculer ou valider le spacing
spacing = validate_or_compute_spacing(user_spacing if not use_uniform_spacing else None)

# Main cylinder
main_cylinder = cq.Workplane("XY").circle(radius_main).extrude(height_main)

# Secondary cylinder
secondary_cylinder = (
    cq.Workplane("XZ")
    .workplane(offset=0.9 * radius_main)
    .center(0, height_main / (num_repetitions + 1))
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

# Appliquer les rotations autour de l'axe Z
secondary_cylinder_rotated = apply_rotations(
    secondary_cylinder, num_rotations, angle_rotation, (0, 0, 0), (0, 0, 1)
)

# Appliquer les répétitions linéaires avec le spacing
secondary_cylinder_repeated = []
for cylinder in secondary_cylinder_rotated:
    secondary_cylinder_repeated.extend(
        apply_linear_repetition(cylinder, num_repetitions, spacing)
    )

# Combiner tous les cylindres secondaires avec le cylindre principal
combined_model = main_cylinder
for cyl in secondary_cylinder_repeated:
    combined_model = combined_model.union(cyl)


# Exporter en fichier .stl 
# exporters.export(combined_model, 'Design (N=0).stl')

# Afficher le modèle final
show_object(combined_model, name="Combined Model")

