// Hybrid unsigned 6x6 multiplier for the balanced-split topology.
//
// The b[1:0] and b[3:2] digit children are carry-predicted approximations
// with independent truth tables; the b[5:4] child is the original exact
// s8862_mul62.  Identical arithmetic to s8862_approx66_hybrid with
// APPROX_MASK=3'b011, except the two approximate instances no longer share
// one module (and therefore one INIT set).
module s8862_approx66_split (
    input  wire [5:0] a,
    input  wire [5:0] b,
    input  wire       high_3,
    input  wire [1:0] stage1_y,
    output wire [11:0] prod,
    output wire       stage1_q6
);

wire [7:0] plow;
wire [7:0] pmid;
wire [7:0] phigh;

s8862_approx62_cp_lo low_block (
    .a(a), .b(b[1:0]), .prod(plow)
);

s8862_approx62_cp_mid middle_block (
    .a(a), .b(b[3:2]), .prod(pmid)
);

s8862_mul62 high_block (
    .a(a), .b(b[5:4]), .high_bit(high_3), .prod(phigh)
);

s8862_comp66_q6 compressor (
    .plow(plow), .pmid(pmid), .phigh(phigh),
    .stage1_x5(b[5]), .stage1_y(stage1_y),
    .prod(prod), .stage1_q6(stage1_q6)
);

endmodule
