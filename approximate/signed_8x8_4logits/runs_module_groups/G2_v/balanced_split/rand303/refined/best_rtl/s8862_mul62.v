// Exact unsigned 6x2 multiplier implemented with LUT6_2 and one CARRY4.
//
// high_bit must be a[5] & b[1].  It is supplied by the parent so that the
// final partial-product bit can share a physical LUT with other top logic.
// Resources inside this module: 5 LUT6_2 + 1 CARRY4.
module s8862_mul62 (
    input  wire [5:0] a,
    input  wire [1:0] b,
    input  wire       high_bit,
    output wire [7:0] prod
);

wire       cin;
wire [3:0] gen;
wire [3:0] prop;
wire [3:0] carry;
wire [3:0] sum;

assign gen[3]  = 1'b0;
assign prop[3] = high_bit;

// Product bits 0 and 1.
LUT6_2 #(
    .INIT(64'h6ac06ac0a0a0a0a0)
) low01_lut (
    .I0(b[0]), .I1(b[1]), .I2(a[0]), .I3(a[1]),
    .I4(1'b1), .I5(1'b1),
    .O5(prod[0]), .O6(prod[1])
);

// O5 is product bit 2; O6 is the carry entering product column 3.
LUT6_2 #(
    .INIT(64'h88008000e6aa4c00)
) bit2_cin_lut (
    .I0(b[0]), .I1(b[1]), .I2(a[0]), .I3(a[1]),
    .I4(a[2]), .I5(1'b1),
    .O5(prod[2]), .O6(cin)
);

// Columns 3..5: each LUT forms the propagate and generate of the two
// same-weight partial-product bits.
LUT6_2 #(
    .INIT(64'h800080006ac06ac0)
) gp3_lut (
    .I0(b[0]), .I1(b[1]), .I2(a[2]), .I3(a[3]),
    .I4(1'b1), .I5(1'b1),
    .O5(prop[0]), .O6(gen[0])
);

LUT6_2 #(
    .INIT(64'h800080006ac06ac0)
) gp4_lut (
    .I0(b[0]), .I1(b[1]), .I2(a[3]), .I3(a[4]),
    .I4(1'b1), .I5(1'b1),
    .O5(prop[1]), .O6(gen[1])
);

LUT6_2 #(
    .INIT(64'h800080006ac06ac0)
) gp5_lut (
    .I0(b[0]), .I1(b[1]), .I2(a[4]), .I3(a[5]),
    .I4(1'b1), .I5(1'b1),
    .O5(prop[2]), .O6(gen[2])
);

CARRY4 product_carry (
    .CO(carry), .O(sum),
    .CI(cin), .CYINIT(1'b0),
    .DI(gen), .S(prop)
);

assign prod[6:3] = sum;
assign prod[7]   = carry[3];

endmodule
