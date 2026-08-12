module approx88 (
    input wire [7:0] a,
    input wire [7:0] b,
    output wire [15:0] prod
);

// ------------------------------------------------------------
// Decomposition
//   a = {ah, al}, ah=a[7:6], al=a[5:0]
//   b = {bh, bl}, bh=b[7:6], bl=b[5:0]
//
//   ll = approx66(al, bl)
//   hl = approx62(bl, ah)
//   lh = approx62(al, bh)
//   hh = ah * bh   (exact 2x2, but implemented only by LUT6/LUT6_2)
//
// Final product:
//   prod â‰? ll + ((hl + lh) << 6) + (hh << 12)
// ------------------------------------------------------------

wire [1:0] ah, bh;
wire [5:0] al, bl;
wire [11:0] ll;
wire [7:0]  hl, lh;
wire        hh_0, hh_1, hh_2;
wire [4:0]  hh;

assign ah = a[7:6];
assign al = a[5:0];
assign bh = b[7:6];
assign bl = b[5:0];

approx66 U_LL (
    .a   (al),
    .b   (bl),
    .prod(ll)
);

accurate62 U_HL (
    .a   (bl),
    .b   (ah),
    .prod(hl)
);

accurate62 U_LH (
    .a   (al),
    .b   (bh),
    .prod(lh)
);

// hh_0 = a[6] & b[6]  (implemented by LUT6)
LUT6 #(
    .INIT(64'h8000_0000_0000_0000)
) LUT_HH0 (
    .I0(a[6]),
    .I1(b[6]),
    .I2(1'b1),
    .I3(1'b1),
    .I4(1'b1),
    .I5(1'b1),
    .O (hh_0)
);

// hh_1 = (a[7]&b[6]) ^ (a[6]&b[7])
// hh_2 = (a[7]&b[6]) & (a[6]&b[7])
LUT6_2 #(
    .INIT(64'h8000800078887888)
) LUT_HH12 (
    .I0(a[7]),
    .I1(b[6]),
    .I2(a[6]),
    .I3(b[7]),
    .I4(1'b1),
    .I5(1'b1),
    .O6(hh_2),
    .O5(hh_1)
);

assign hh = {a[7], b[7], hh_2, hh_1, hh_0};

comp88 U_COMP (
    .hh   (hh),
    .hl   (hl),
    .lh   (lh),
    .ll   (ll),
    .prod (prod)
);

endmodule
