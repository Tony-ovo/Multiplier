module tcasi_unsigned_core (
    input  wire [7:0]  a,
    input  wire [7:0]  b,
    output wire [15:0] prod8
);
    ac_4444 core(
        .a(a),
        .b(b),
        .prod8(prod8)
    );
endmodule
