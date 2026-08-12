// Area-oriented approximate signed 8x8 multiplier.
// AL is quantized to multiples of 16 and the lower LL carry segment is cut.
// Resources: 29 LUT6_2 + 5 CARRY4.

module s8862_signed_finish (
    input  wire signed [7:0]  a,
    input  wire signed [7:0]  b,
    input  wire        [11:0] ll,
    input  wire               stage1_q6,
    output wire signed [15:0] prod
);
wire [9:0] h0;
wire [9:0] h1;
wire [9:0] h2;
assign h0 = {4'b0000, ll[11:6]};
s8862_mac_u6_s2 stage1 (
    .acc(h0), .x(b[5:0]), .y(a[7:6]),
    .shared_q6(stage1_q6), .sum(h1)
);
s8862_mac_s8_s2 stage2 (
    .acc(h1), .x(a), .y(b[7:6]), .sum(h2)
);
assign prod = {h2, ll[5:0]};
endmodule

(* use_dsp = "no" *)
module signed88_approx_area (
    input  wire signed [7:0]  a,
    input  wire signed [7:0]  b,
    output wire signed [15:0] prod
);
wire [7:0] plow;
wire [7:0] pmid;
wire [7:0] phigh;
wire [11:0] ll;
wire stage1_q6;

s8862_approx62_q16 low_block (
    .a(a[5:0]), .b(b[1:0]), .prod(plow)
);
s8862_approx62_q16 middle_block (
    .a(a[5:0]), .b(b[3:2]), .prod(pmid)
);
s8862_approx62_q16 high_block (
    .a(a[5:0]), .b(b[5:4]), .prod(phigh)
);
s8862_comp66_q16 compressor (
    .plow(plow), .pmid(pmid), .phigh(phigh),
    .stage1_x5(b[5]), .stage1_y(a[7:6]),
    .prod(ll), .stage1_q6(stage1_q6)
);
s8862_signed_finish finish (
    .a(a), .b(b), .ll(ll), .stage1_q6(stage1_q6), .prod(prod)
);
endmodule
