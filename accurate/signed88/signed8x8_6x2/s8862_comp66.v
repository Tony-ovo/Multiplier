// Compress three unsigned 6x2 products into an exact unsigned 6x6 product:
//
//   prod = plow + (pmid << 2) + (phigh << 4)
//
// Resources: 9 LUT6_2 + 2 CARRY4.
module s8862_comp66 (
    input  wire [7:0] plow,
    input  wire [7:0] pmid,
    input  wire [7:0] phigh,
    output wire [11:0] prod
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

// Columns 2 and 3 of plow + (pmid << 2).
LUT6_2 #(
    .INIT(64'h936c936c5a5a5a5a)
) low_sum_lut (
    .I0(plow[2]), .I1(plow[3]),
    .I2(pmid[0]), .I3(pmid[1]),
    .I4(1'b1), .I5(1'b1),
    .O5(prod[2]), .O6(prod[3])
);

// O5 is the carry entering global product column 4. O6 is unused.
LUT6_2 #(
    .INIT(64'h00000000ec80ec80)
) low_carry_lut (
    .I0(plow[2]), .I1(plow[3]),
    .I2(pmid[0]), .I3(pmid[1]),
    .I4(1'b1), .I5(1'b1),
    .O5(cin), .O6(low_unused)
);

// First aligned three-row compressor column (global product column 4).
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

// Tail propagate for global product column 10. O5 is unused.
wire tail_unused;
LUT6_2 #(
    .INIT(64'h6666666600000000)
) tail_lut (
    .I0(a_row[6]), .I1(g[5]),
    .I2(1'b1), .I3(1'b1), .I4(1'b1), .I5(1'b1),
    .O5(tail_unused), .O6(p[6])
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
