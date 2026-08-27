// Area/speed-oriented compressor specialized for three
// s8862_approx62_q16 products.
//
// Every child has bits [3:0]=0.  Therefore product bits [3:0], the generic
// compressor's low-sum LUT, and its low-carry LUT all disappear.  The
// first four-bit carry segment is also removed: prod[7:4] takes the
// carry-save propagate row directly, and the remaining upper CARRY4 starts
// with CI=0.  This deliberate carry cut costs little average accuracy for
// the already-quantized operands while saving one more carry primitive and
// breaking the LL carry path into a single four-bit segment.
// Resources: 7 LUT6_2 + 1 CARRY4.
module s8862_comp66_q16 (
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
wire [7:0] p;
wire [7:0] g;

assign a_row = phigh;
assign b_row = pmid[7:2];
assign c_row = {2'b00, plow[7:4]};
assign prod[3:0] = 4'b0000;
assign p[7] = a_row[7];
assign g[7:6] = 2'b00;

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

LUT6_2 #(
    .INIT(64'h00ffff00d0d0d0d0)
) tail_q6_lut (
    .I0(stage1_x5), .I1(stage1_y[0]), .I2(stage1_y[1]),
    .I3(a_row[6]), .I4(g[5]), .I5(1'b1),
    .O5(stage1_q6), .O6(p[6])
);

wire [7:0] sum;

assign sum[3:0] = p[3:0];

wire [3:0] carry;
CARRY4 carry_1 (
    .CO(carry), .O(sum[7:4]),
    .CI(1'b0), .CYINIT(1'b0),
    .DI(g[6:3]), .S(p[7:4])
);

assign prod[11:4] = sum;

endmodule
