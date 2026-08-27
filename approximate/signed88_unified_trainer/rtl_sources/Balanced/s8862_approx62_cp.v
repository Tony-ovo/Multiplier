// Carry-predicted unsigned 6x2 multiplier.
//
// The two-bit digit b is exact for 0, 1 and 2.  Only b=3 needs the
// addition a + (a << 1), so only that case is approximated.  The four
// LUT6_2 cells use overlapping local windows of a to predict the carries
// crossing bit-pair boundaries.  For b=3 the two boundary predictors are:
//
//   r2 = a[1] & (a[2] ^ a[3])
//   r4 = a[3]
//
// (Both are implicitly gated by b[1]&b[0] in the LUT truth tables.)
//
// Exhaustive unsigned 6x2 characteristics (256 input pairs):
//   ER   = 28/256 = 10.9375%
//   MAE  = 1.125
//   WCE  = 16
//   bias = 0
//
// Resources: 4 LUT6_2, no CARRY4.
module s8862_approx62_cp (
    input  wire [5:0] a,
    input  wire [1:0] b,
    output wire [7:0] prod
);

// Bits 0 and 1 are exact for every digit.
LUT6_2 #(
    .INIT(64'h6ac06ac0a0a0a0a0)
) low01_lut (
    .I0(b[0]), .I1(b[1]), .I2(a[0]), .I3(a[1]),
    .I4(1'b1), .I5(1'b1),
    .O5(prod[0]), .O6(prod[1])
);

// For b=3, bits [3:2] use the local window a[3:1].  The omitted a[0]
// is equivalent to predicting the carry entering product bit 2.
LUT6_2 #(
    .INIT(64'he62a4c006a40eac0)
) pair23_lut (
    .I0(b[0]), .I1(b[1]), .I2(a[1]), .I3(a[2]),
    .I4(a[3]), .I5(1'b1),
    .O5(prod[2]), .O6(prod[3])
);

// For b=3, bits [5:4] predict the incoming carry from a[5:3].
LUT6_2 #(
    .INIT(64'he62a4c80ea40ea40)
) pair45_lut (
    .I0(b[0]), .I1(b[1]), .I2(a[3]), .I3(a[4]),
    .I4(a[5]), .I5(1'b1),
    .O5(prod[4]), .O6(prod[5])
);

// The two most-significant bits reuse the same a[5:3] window.
LUT6_2 #(
    .INIT(64'h88800000444c8000)
) pair67_lut (
    .I0(b[0]), .I1(b[1]), .I2(a[3]), .I3(a[4]),
    .I4(a[5]), .I5(1'b1),
    .O5(prod[6]), .O6(prod[7])
);

endmodule
