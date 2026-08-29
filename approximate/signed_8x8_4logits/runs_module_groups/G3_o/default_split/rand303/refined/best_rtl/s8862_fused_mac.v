// Fused accumulator stages used by signed88_6x2.
//
// A two-bit two's-complement digit y has value y[0] - 2*y[1]. For each bit,
// form a complemented operand q and add acc + q + y[1]. Every generic LUT6_2
// simultaneously creates O5=q and O6=q^acc, which directly drive CARRY4 DI/S.

// -----------------------------------------------------------------------------
// Stage 1: sum = acc + signed(y) * unsigned(x) (mod 2^10).
// x is six-bit unsigned. shared_q6 comes from a top shared LUT.
// Resources: 7 LUT6_2 + 2 CARRY4.
// -----------------------------------------------------------------------------
module s8862_mac_u6_s2 (
    input  wire [9:0] acc,
    input  wire [5:0] x,
    input  wire [1:0] y,
    input  wire       shared_q6,
    output wire [9:0] sum
);

wire [5:0] q;
wire [5:0] s;

genvar i;
generate
    for (i = 0; i < 6; i = i + 1) begin : stage1_cells
        if (i == 0) begin : first_bit
            LUT6_2 #(
                .INIT(64'hd973268c268c268c)
            ) lut_cell (
                .I0(y[0]), .I1(y[1]), .I2(1'b0), .I3(x[0]),
                .I4(acc[0]), .I5(1'b1),
                .O5(q[0]), .O6(s[0])
            );
        end else begin : other_bits
            LUT6_2 #(
                .INIT(64'hd973268c268c268c)
            ) lut_cell (
                .I0(y[0]), .I1(y[1]), .I2(x[i-1]), .I3(x[i]),
                .I4(acc[i]), .I5(1'b1),
                .O5(q[i]), .O6(s[i])
            );
        end
    end
endgenerate

wire [3:0] carry0;
wire [3:0] carry1;
wire [7:0] low_sum;

CARRY4 carry_0 (
    .CO(carry0), .O(low_sum[3:0]),
    .CI(1'b0), .CYINIT(y[1]),
    .DI(q[3:0]), .S(s[3:0])
);

// acc[6]=acc[7]=0. q7 is pure sign extension y[1].
CARRY4 carry_1 (
    .CO(carry1), .O(low_sum[7:4]),
    .CI(carry0[3]), .CYINIT(1'b0),
    .DI({y[1], shared_q6, q[5], q[4]}),
    .S ({y[1], shared_q6, s[5], s[4]})
);

assign sum[7:0] = low_sum;

// Bits 8 and 9 are the addition of sign-extension q=y1 and the carry from
// bit 7. One LUT replaces a third CARRY4.
LUT6_2 #(
    .INIT(64'h2222222266666666)
) top_bits_lut (
    .I0(y[1]), .I1(carry1[3]),
    .I2(1'b1), .I3(1'b1), .I4(1'b1), .I5(1'b1),
    .O5(sum[8]), .O6(sum[9])
);

endmodule


// -----------------------------------------------------------------------------
// Stage 2: sum = acc + signed(y) * signed(x) (mod 2^10).
// x is signed eight-bit; y is a two-bit two's-complement digit.
// Resources: 9 LUT6_2 + 2 CARRY4.
// -----------------------------------------------------------------------------
module s8862_mac_s8_s2 (
    input  wire [9:0]        acc,
    input  wire signed [7:0] x,
    input  wire [1:0]        y,
    output wire [9:0]        sum
);

wire [6:0] q;
wire [6:0] s;

genvar j;
generate
    for (j = 0; j < 7; j = j + 1) begin : stage2_cells
        if (j == 0) begin : first_bit
            LUT6_2 #(
                .INIT(64'hd973268c268c268c)
            ) lut_cell (
                .I0(y[0]), .I1(y[1]), .I2(1'b0), .I3(x[0]),
                .I4(acc[0]), .I5(1'b1),
                .O5(q[0]), .O6(s[0])
            );
        end else begin : other_bits
            LUT6_2 #(
                .INIT(64'hd973268c268c268c)
            ) lut_cell (
                .I0(y[0]), .I1(y[1]), .I2(x[j-1]), .I3(x[j]),
                .I4(acc[j]), .I5(1'b1),
                .O5(q[j]), .O6(s[j])
            );
        end
    end
endgenerate

wire q_sign;
wire s7;

// O5 supplies the sign-extension row for bits 8/9; O6 supplies S7. At bit 7,
// DI=acc7 is carry-equivalent to DI=q7 when S7=q7^acc7.
LUT6_2 #(
    .INIT(64'hd973268c22cc22cc)
) bit7_sign_lut (
    .I0(y[0]), .I1(y[1]), .I2(x[6]), .I3(x[7]),
    .I4(acc[7]), .I5(1'b1),
    .O5(q_sign), .O6(s7)
);

wire [3:0] carry0;
wire [3:0] carry1;
wire [7:0] low_sum;

CARRY4 carry_0 (
    .CO(carry0), .O(low_sum[3:0]),
    .CI(1'b0), .CYINIT(y[1]),
    .DI(q[3:0]), .S(s[3:0])
);

CARRY4 carry_1 (
    .CO(carry1), .O(low_sum[7:4]),
    .CI(carry0[3]), .CYINIT(1'b0),
    .DI({acc[7], q[6], q[5], q[4]}),
    .S ({s7,     s[6], s[5], s[4]})
);

assign sum[7:0] = low_sum;

// O5 is bit 8. O6 includes the bit-8 majority carry and generates bit 9.
LUT6_2 #(
    .INIT(64'hbd42bd4296969696)
) top_bits_lut (
    .I0(q_sign), .I1(acc[8]), .I2(carry1[3]), .I3(acc[9]),
    .I4(1'b1), .I5(1'b1),
    .O5(sum[8]), .O6(sum[9])
);

endmodule
