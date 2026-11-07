// Dimensions in millimetres; user-authorized fit-selected shaft, original12x28grip.
total_length = 42.5;
shaft_length = 14.5;
shaft_diameter = 2.3;
grip_diameter = 12;
grip_length = total_length - shaft_length;
$fn = 128;
union() {
    cylinder(h = grip_length, d = grip_diameter);
    translate([0, 0, grip_length - 0.01])
        cylinder(h = shaft_length + 0.01, d = shaft_diameter);
}
