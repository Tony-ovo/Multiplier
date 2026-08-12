// Standard two-input/one-output wrapper for the Balanced implementation.
(* use_dsp = "no" *)
module s88_top (
    input  wire signed [7:0]  a,
    input  wire signed [7:0]  b,
    output wire signed [15:0] prod
);

signed88_approx_balanced multiplier (
    .a(a),
    .b(b),
    .prod(prod)
);

endmodule
