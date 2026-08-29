// Carry-predicted unsigned 6x2 multipliers with per-segment truth tables.
//
// Three INIT-decoupled copies of the original s8862_approx62_cp used by
// Default/Fast.  The parent always instantiated three physical copies (for
// b[1:0], b[3:2], b[5:4]); giving each copy its own module lets the trainer
// patch their INITs independently at zero extra LUT cost.
// Resources per module: 4 LUT6_2, no CARRY4.

module s8862_approx62_cp_lo (
    input  wire [5:0] a,
    input  wire [1:0] b,
    output wire [7:0] prod
);

LUT6_2 #(
    .INIT(64'h62C06AC028A0A0A0)
) low01_lut (
    .I0(b[0]), .I1(b[1]), .I2(a[0]), .I3(a[1]),
    .I4(1'b1), .I5(1'b1),
    .O5(prod[0]), .O6(prod[1])
);

LUT6_2 #(
    .INIT(64'hE6224480EA406AC0)
) pair23_lut (
    .I0(b[0]), .I1(b[1]), .I2(a[1]), .I3(a[2]),
    .I4(a[3]), .I5(1'b1),
    .O5(prod[2]), .O6(prod[3])
);

LUT6_2 #(
    .INIT(64'hE6AA4C80E248EAC0)
) pair45_lut (
    .I0(b[0]), .I1(b[1]), .I2(a[3]), .I3(a[4]),
    .I4(a[5]), .I5(1'b1),
    .O5(prod[4]), .O6(prod[5])
);

LUT6_2 #(
    .INIT(64'h8088000044448880)
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
    .INIT(64'h6AC06AC02020A0A0)
) low01_lut (
    .I0(b[0]), .I1(b[1]), .I2(a[0]), .I3(a[1]),
    .I4(1'b1), .I5(1'b1),
    .O5(prod[0]), .O6(prod[1])
);

LUT6_2 #(
    .INIT(64'h6622CC006240EAC0)
) pair23_lut (
    .I0(b[0]), .I1(b[1]), .I2(a[1]), .I3(a[2]),
    .I4(a[3]), .I5(1'b1),
    .O5(prod[2]), .O6(prod[3])
);

LUT6_2 #(
    .INIT(64'h662ACC8062486AC0)
) pair45_lut (
    .I0(b[0]), .I1(b[1]), .I2(a[3]), .I3(a[4]),
    .I4(a[5]), .I5(1'b1),
    .O5(prod[4]), .O6(prod[5])
);

LUT6_2 #(
    .INIT(64'h80800000CC4C8000)
) pair67_lut (
    .I0(b[0]), .I1(b[1]), .I2(a[3]), .I3(a[4]),
    .I4(a[5]), .I5(1'b1),
    .O5(prod[6]), .O6(prod[7])
);

endmodule

module s8862_approx62_cp_hi (
    input  wire [5:0] a,
    input  wire [1:0] b,
    output wire [7:0] prod
);

LUT6_2 #(
    .INIT(64'h62C06AC0A020A0A0)
) low01_lut (
    .I0(b[0]), .I1(b[1]), .I2(a[0]), .I3(a[1]),
    .I4(1'b1), .I5(1'b1),
    .O5(prod[0]), .O6(prod[1])
);

LUT6_2 #(
    .INIT(64'hE622C400EA40EAC0)
) pair23_lut (
    .I0(b[0]), .I1(b[1]), .I2(a[1]), .I3(a[2]),
    .I4(a[3]), .I5(1'b1),
    .O5(prod[2]), .O6(prod[3])
);

LUT6_2 #(
    .INIT(64'hE6224C00E248EAC0)
) pair45_lut (
    .I0(b[0]), .I1(b[1]), .I2(a[3]), .I3(a[4]),
    .I4(a[5]), .I5(1'b1),
    .O5(prod[4]), .O6(prod[5])
);

LUT6_2 #(
    .INIT(64'h888000004C4C8000)
) pair67_lut (
    .I0(b[0]), .I1(b[1]), .I2(a[3]), .I3(a[4]),
    .I4(a[5]), .I5(1'b1),
    .O5(prod[6]), .O6(prod[7])
);

endmodule
