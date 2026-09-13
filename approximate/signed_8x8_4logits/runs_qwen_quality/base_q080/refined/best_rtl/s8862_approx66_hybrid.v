// Hybrid unsigned 6x6 multiplier used by the signed approximate tops.
//
// APPROX_MASK bit k selects the carry-predicted 6x2 child for digit
// b[2*k +: 2].  A zero selects the original exact s8862_mul62 child.
// The following useful configurations are instantiated by the top-level file:
//   3'b001: quality   (only the least-significant child is approximate)
//   3'b011: balanced  (low and middle children are approximate)
//   3'b111: fast      (all three children are approximate)
module s8862_approx66_hybrid #(
    parameter [2:0] APPROX_MASK = 3'b111
) (
    input  wire [5:0] a,
    input  wire [5:0] b,
    input  wire       high_1,
    input  wire       high_2,
    input  wire       high_3,
    input  wire [1:0] stage1_y,
    output wire [11:0] prod,
    output wire       stage1_q6
);

wire [7:0] plow;
wire [7:0] pmid;
wire [7:0] phigh;

generate
    if (APPROX_MASK[0]) begin : gen_approx_low
        s8862_approx62_cp low_block (
            .a(a), .b(b[1:0]), .prod(plow)
        );
    end else begin : gen_exact_low
        s8862_mul62 low_block (
            .a(a), .b(b[1:0]), .high_bit(high_1), .prod(plow)
        );
    end

    if (APPROX_MASK[1]) begin : gen_approx_mid
        s8862_approx62_cp middle_block (
            .a(a), .b(b[3:2]), .prod(pmid)
        );
    end else begin : gen_exact_mid
        s8862_mul62 middle_block (
            .a(a), .b(b[3:2]), .high_bit(high_2), .prod(pmid)
        );
    end

    if (APPROX_MASK[2]) begin : gen_approx_high
        s8862_approx62_cp high_block (
            .a(a), .b(b[5:4]), .prod(phigh)
        );
    end else begin : gen_exact_high
        s8862_mul62 high_block (
            .a(a), .b(b[5:4]), .high_bit(high_3), .prod(phigh)
        );
    end
endgenerate

s8862_comp66_q6 compressor (
    .plow(plow), .pmid(pmid), .phigh(phigh),
    .stage1_x5(b[5]), .stage1_y(stage1_y),
    .prod(prod), .stage1_q6(stage1_q6)
);

endmodule
