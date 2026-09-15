// Exact compression of three (possibly approximate) unsigned 6x2 products:
//
//   prod = plow + (pmid << 2) + (phigh << 4)  (modulo the 12-bit range)
//
// The compressor arithmetic itself is exact.  Its tail LUT also supplies
// stage1_q6 for the signed upper-part accumulator at no extra LUT cost.
// Resources: 9 LUT6_2 + 2 CARRY4.
module s8862_comp66_q6 (
    input  wire [7:0] plow,
    input  wire [7:0] pmid,
    input  wire [7:0] phigh,
    input  wire       stage1_x5,
    input  wire [1:0] stage1_y,
    output wire [11:0] prod,
    output wire       stage1_q6
);

wire [7:0] a_row;
wire [5:0] b_row;
wire [5:0] c_row;

assign a_row = phigh;
assign b_row = pmid[7:2];
assign c_row = {2'b00, plow[7:4]};
assign prod[1:0] = plow[1:0];

wire       cin;
wire       low_unused;
wire [7:0] p;
wire [7:0] g;

assign p[7]   = a_row[7];
assign g[7:6] = 2'b00;

LUT6_2 #(
    .INIT(64'h936c936c5a5a5a5a)
) low_sum_lut (
    .I0(plow[2]), .I1(plow[3]),
    .I2(pmid[0]), .I3(pmid[1]),
    .I4(1'b1), .I5(1'b1),
    .O5(prod[2]), .O6(prod[3])
);

LUT6_2 #(
    .INIT(64'h00000000ec80ec80)
) low_carry_lut (
    .I0(plow[2]), .I1(plow[3]),
    .I2(pmid[0]), .I3(pmid[1]),
    .I4(1'b1), .I5(1'b1),
    .O5(cin), .O6(low_unused)
);

LUT6_2 #(
    .INIT(64'h96969696e8e8e8e8)
) csa0_lut (
    .I0(c_row[0]), .I1(b_row[0]), .I2(a_row[0]),
    .I3(1'b1), .I4(1'b1), .I5(1'b1),
    .O5(g[0]), .O6(p[0])
);

genvar i;
generate
    for (i = 1; i < 6; i = i + 1) begin : csa_gp
        LUT6_2 #(
            .INIT(64'h69966996e8e8e8e8)
        ) csa_lut (
            .I0(c_row[i]), .I1(b_row[i]), .I2(a_row[i]), .I3(g[i-1]),
            .I4(1'b1), .I5(1'b1),
            .O5(g[i]), .O6(p[i])
        );
    end
endgenerate

// O5 = stage1_y[1] & (stage1_y[0] | ~stage1_x5)
// O6 = a_row[6] ^ g[5]
//
// Folding q6 here removes the separate shared-high/q6 LUT used by the exact
// top level.  O5 uses the lower 32 INIT bits and O6 the upper 32 bits.
LUT6_2 #(
    .INIT(64'h00ffff00d0d0d0d0)
) tail_q6_lut (
    .I0(stage1_x5), .I1(stage1_y[0]), .I2(stage1_y[1]),
    .I3(a_row[6]), .I4(g[5]), .I5(1'b1),
    .O5(stage1_q6), .O6(p[6])
);

wire [8:0] carry;
wire [7:0] sum;
assign carry[0] = cin;

CARRY4 carry_0 (
    .CO(carry[4:1]), .O(sum[3:0]),
    .CI(carry[0]), .CYINIT(1'b0),
    .DI({g[2:0], 1'b0}), .S(p[3:0])
);

CARRY4 carry_1 (
    .CO(carry[8:5]), .O(sum[7:4]),
    .CI(carry[4]), .CYINIT(1'b0),
    .DI(g[6:3]), .S(p[7:4])
);

assign prod[11:4] = sum;

endmodule
