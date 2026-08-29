// Carry-predicted unsigned 6x2 multipliers with per-segment truth tables.
//
// These are two INIT-decoupled copies of the original s8862_approx62_cp.
// The balanced topology always instantiated two physical copies (for the
// b[1:0] and b[3:2] digits); giving each copy its own module lets the
// trainer patch their INITs independently at zero hardware cost.
// Resources per module: 4 LUT6_2, no CARRY4.

module s8862_approx62_cp_lo (
    input  wire [5:0] a,
    input  wire [1:0] b,
    output wire [7:0] prod
);

LUT6_2 #(
    .INIT(64'h6AC06AC0A0A0A0A0)
) low01_lut (
    .I0(b[0]), .I1(b[1]), .I2(a[0]), .I3(a[1]),
    .I4(1'b1), .I5(1'b1),
    .O5(prod[0]), .O6(prod[1])
);

LUT6_2 #(
    .INIT(64'hE62ACC00EA406A40)
) pair23_lut (
    .I0(b[0]), .I1(b[1]), .I2(a[1]), .I3(a[2]),
    .I4(a[3]), .I5(1'b1),
    .O5(prod[2]), .O6(prod[3])
);

LUT6_2 #(
    .INIT(64'hE62A4C00EA40EAC0)
) pair45_lut (
    .I0(b[0]), .I1(b[1]), .I2(a[3]), .I3(a[4]),
    .I4(a[5]), .I5(1'b1),
    .O5(prod[4]), .O6(prod[5])
);

LUT6_2 #(
    .INIT(64'h88800000444C8000)
) pair67_lut (
    .I0(b[0]), .I1(b[1]), .I2(a[3]), .I3(a[4]),
    .I4(a[5]), .I5(1'b1),
    .O5(prod[6]), .O6(prod[7])
);

endmodule

module s8862_approx62_cp_mid (
    input  wire [5:0] a,
    input  wire [1:0] b,
    output wire [7:0] prod
);

LUT6_2 #(
    .INIT(64'h6AC06AC0A0A0A0A0)
) low01_lut (
    .I0(b[0]), .I1(b[1]), .I2(a[0]), .I3(a[1]),
    .I4(1'b1), .I5(1'b1),
    .O5(prod[0]), .O6(prod[1])
);

LUT6_2 #(
    .INIT(64'hE622CC00EA40EAC0)
) pair23_lut (
    .I0(b[0]), .I1(b[1]), .I2(a[1]), .I3(a[2]),
    .I4(a[3]), .I5(1'b1),
    .O5(prod[2]), .O6(prod[3])
);

LUT6_2 #(
    .INIT(64'hE62A4C80EA40EA40)
) pair45_lut (
    .I0(b[0]), .I1(b[1]), .I2(a[3]), .I3(a[4]),
    .I4(a[5]), .I5(1'b1),
    .O5(prod[4]), .O6(prod[5])
);

LUT6_2 #(
    .INIT(64'h88800000444C8000)
) pair67_lut (
    .I0(b[0]), .I1(b[1]), .I2(a[3]), .I3(a[4]),
    .I4(a[5]), .I5(1'b1),
    .O5(prod[6]), .O6(prod[7])
);

endmodule
