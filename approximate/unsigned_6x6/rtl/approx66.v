module approx66 (
    input wire [5:0] a,
    input wire [5:0] b,
    output wire [11:0] prod
);

wire [7:0] plow;
wire [7:0] pmid;
wire [7:0] phigh;

// Three approximate 6x2 multipliers
approx62 U_LOW (
    .a(a),
    .b(b[1:0]),
    .prod(plow)
);

approx62 U_MID (
    .a(a),
    .b(b[3:2]),
    .prod(pmid)
);

approx62 U_HIGH (
    .a(a),
    .b(b[5:4]),
    .prod(phigh)
);

// One hybrid compressor:
// - low overlap columns (bit2, bit3) are approximated
// - higher columns still use LUT + CARRY4 compression
comp66 U_COMP (
    .plow(plow),
    .pmid(pmid),
    .phigh(phigh),
    .prod(prod)
);

endmodule


