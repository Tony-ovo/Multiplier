module s88_top (
    input  wire signed [7:0]  a,
    input  wire signed [7:0]  b,
    output wire signed [15:0] prod
);
    wire sign;
    wire [7:0] mag_a;
    wire [7:0] mag_b;
    wire [15:0] mag_prod;
    wire signed [16:0] signed_prod_ext;

    assign sign = a[7] ^ b[7];
    assign mag_a = a[7] ? (~a + 8'd1) : a;
    assign mag_b = b[7] ? (~b + 8'd1) : b;

    tcasi_unsigned_core core(
        .a(mag_a),
        .b(mag_b),
        .prod8(mag_prod)
    );

    // The unsigned core computes magnitudes; restore the signed product in
    // one wider intermediate before truncating to the 16-bit result.
    assign signed_prod_ext = sign
        ? -$signed({1'b0, mag_prod})
        :  $signed({1'b0, mag_prod});
    assign prod = signed_prod_ext[15:0];
endmodule
