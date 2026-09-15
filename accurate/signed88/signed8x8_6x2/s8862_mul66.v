// Exact unsigned 6x6 multiplier built from three exact unsigned 6x2 blocks.
// high_1/2/3 must be a[5]&b[1], a[5]&b[3], a[5]&b[5].
// Resources inside this module: 24 LUT6_2 + 5 CARRY4.
module s8862_mul66 (
    input  wire [5:0] a,
    input  wire [5:0] b,
    input  wire       high_1,
    input  wire       high_2,
    input  wire       high_3,
    output wire [11:0] prod
);

wire [7:0] plow;
wire [7:0] pmid;
wire [7:0] phigh;

s8862_mul62 low_block (
    .a(a), .b(b[1:0]), .high_bit(high_1), .prod(plow)
);

s8862_mul62 middle_block (
    .a(a), .b(b[3:2]), .high_bit(high_2), .prod(pmid)
);

s8862_mul62 high_block (
    .a(a), .b(b[5:4]), .high_bit(high_3), .prod(phigh)
);

s8862_comp66 compressor (
    .plow(plow), .pmid(pmid), .phigh(phigh), .prod(prod)
);

endmodule
