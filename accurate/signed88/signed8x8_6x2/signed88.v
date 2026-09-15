// Exact two's-complement signed 8x8 multiplier.
//
// Low 6x6: three unsigned 6x2 blocks plus a compressor.
// High ten bits: two fused signed-digit accumulator stages.
//
// Expected explicit resources:
//   42 LUT6_2 + 9 CARRY4
//   no LUT6, DSP, register, or RAM.
(* use_dsp = "no" *)
module signed88_6x2 (
    input  wire signed [7:0]  a,
    input  wire signed [7:0]  b,
    output wire signed [15:0] prod
);

wire [5:0] al;
wire [5:0] bl;
wire [11:0] ll;
wire [9:0] h0;
wire [9:0] h1;
wire [9:0] h2;

wire ll_high_1;
wire ll_high_2;
wire ll_high_3;
wire stage1_q6;

assign al = a[5:0];
assign bl = b[5:0];

// Highest partial-product bits for the first two 6x2 children:
//   O5 = al5 & bl1
//   O6 = al5 & bl3
LUT6_2 #(
    .INIT(64'ha0a0a0a088888888)
) shared_high12_lut (
    .I0(al[5]), .I1(bl[1]), .I2(bl[3]),
    .I3(1'b1), .I4(1'b1), .I5(1'b1),
    .O5(ll_high_1), .O6(ll_high_2)
);

// O5 is the third 6x2 highest bit. O6 is the first fused MAC's q6:
//   ll_high_3 = al5 & bl5
//   stage1_q6 = a7 & (a6 | ~bl5)
LUT6_2 #(
    .INIT(64'hf300f30088888888)
) shared_high3_q6_lut (
    .I0(al[5]), .I1(bl[5]), .I2(a[6]), .I3(a[7]),
    .I4(1'b1), .I5(1'b1),
    .O5(ll_high_3), .O6(stage1_q6)
);

s8862_mul66 low_multiplier (
    .a(al), .b(bl),
    .high_1(ll_high_1), .high_2(ll_high_2), .high_3(ll_high_3),
    .prod(ll)
);

// A=AL+64*AH, B=BL+64*BH and
// A*B = AL*BL + 64*(AH*BL + A*BH).
assign h0 = {4'b0000, ll[11:6]};

s8862_mac_u6_s2 stage1 (
    .acc(h0), .x(bl), .y(a[7:6]),
    .shared_q6(stage1_q6), .sum(h1)
);

s8862_mac_s8_s2 stage2 (
    .acc(h1), .x(a), .y(b[7:6]), .sum(h2)
);

assign prod = {h2, ll[5:0]};

endmodule
