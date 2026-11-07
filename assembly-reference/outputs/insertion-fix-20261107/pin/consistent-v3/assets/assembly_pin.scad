// Isolated v3: fit-first pin for the centered completed connector target.
// Dimensions in millimetres; receiving CAD remains unchanged.
total_length = 42;
shaft_length = 14;
shaft_diameter = 2.3;
grip_diameter = 12;
grip_length = 28;
$fn = 128;
union() {
    cylinder(h = grip_length, d = grip_diameter);
    translate([0, 0, grip_length - 0.01])
        cylinder(h = shaft_length + 0.01, d = shaft_diameter);
}
