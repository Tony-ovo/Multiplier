// Area-oriented unsigned 6x2 approximation.
//
// Quantize a to the nearest multiple of 16, with half values rounded up:
//
//   aq = 16 * ((a + 8) >> 4)
//   prod ~= aq * b
//
// aq can be 64 when a>=56; the full eight-bit product still represents
// 64*3=192.  Since prod[3:0] is always zero, only two LUT6_2 cells remain.
// Resources: 2 LUT6_2, no CARRY4.
module s8862_approx62_q16 (
    input  wire [5:0] a,
    input  wire [1:0] b,
    output wire [7:0] prod
);

assign prod[3:0] = 4'b0000;

LUT6_2 #(
    .INIT(64'h066aacc00aa00aa0)
) high45_lut (
    .I0(b[0]), .I1(b[1]), .I2(a[3]), .I3(a[4]),
    .I4(a[5]), .I5(1'b1),
    .O5(prod[4]), .O6(prod[5])
);

LUT6_2 #(
    .INIT(64'hc8800000a44cc000)
) high67_lut (
    .I0(b[0]), .I1(b[1]), .I2(a[3]), .I3(a[4]),
    .I4(a[5]), .I5(1'b1),
    .O5(prod[6]), .O6(prod[7])
);

endmodule
