//accurate--comp88

module comp88 (
    input wire [4:0]  hh,
    input wire [7:0]  hl,
    input wire [7:0]  lh,
    input wire [11:0] ll,
    output wire [15:0] prod
);

wire [8:0] a_reg;
wire [7:0] b_reg;
wire [7:0] c_reg;
wire [11:0] p;
wire [11:0] g;
wire [12:0] c_i;
wire [11:0] sum;

assign a_reg[5:0] = ll[11:6];
assign a_reg[8:6] = hh[2:0];
assign b_reg[7:0] = lh[7:0];
assign c_reg[7:0] = hl[7:0];

assign prod[5:0] = ll[5:0];
assign g[11:9] = 3'b000;
assign p[11:9] = {2'b00, g[8]};
assign c_i[0]  = 1'b0;

LUT6_2 #(
    .INIT(64'h96969696E8E8E8E8)
) LUT_GP0 (
    .I0(c_reg[0]),
    .I1(b_reg[0]),
    .I2(a_reg[0]),
    .I3(1'b1),
    .I4(1'b1),
    .I5(1'b1),
    .O6(p[0]),
    .O5(g[0])
);

genvar j;
generate
    for (j = 1; j < 8; j = j + 1) begin : GP88
        LUT6_2 #(
            .INIT(64'h69966996E8E8E8E8)
        ) LUT_GP (
            .I0(c_reg[j]),
            .I1(b_reg[j]),
            .I2(a_reg[j]),
            .I3(g[j-1]),
            .I4(1'b1),
            .I5(1'b1),
            .O6(p[j]),
            .O5(g[j])
        );
    end
endgenerate

LUT6_2 #(
    .INIT(64'h96669666A000A000)
) LUT_GP8 (
    .I0(a_reg[8]),
    .I1(g[7]),
    .I2(hh[4]),
    .I3(hh[3]),
    .I4(1'b1),
    .I5(1'b1),
    .O6(p[8]),
    .O5(g[8])
);

CARRY4 CARRY4_0 (
    .CO     (c_i[4:1]),
    .O      (sum[3:0]),
    .CI     (c_i[0]),
    .CYINIT (1'b0),
    .DI     ({g[2:0], 1'b0}),
    .S      (p[3:0])
);

CARRY4 CARRY4_1 (
    .CO     (c_i[8:5]),
    .O      (sum[7:4]),
    .CI     (c_i[4]),
    .CYINIT (1'b0),
    .DI     (g[6:3]),
    .S      (p[7:4])
);

CARRY4 CARRY4_2 (
    .CO     (c_i[12:9]),
    .O      (sum[11:8]),
    .CI     (c_i[8]),
    .CYINIT (1'b0),
    .DI     (g[10:7]),
    .S      (p[11:8])
);

assign prod[15:6] = sum[9:0];

endmodule




// module comp88 (
//     input wire [4:0]  hh,
//     input wire [7:0]  hl,
//     input wire [7:0]  lh,
//     input wire [11:0] ll,
//     output wire [15:0] prod
// );

// // ------------------------------------------------------------
// // Column alignment for final 8x8 compression:
// // bit0~5 : ll[5:0]
// // bit6   : ll[6]  + hl[0] + lh[0]
// // bit7   : ll[7]  + hl[1] + lh[1]
// // bit8   : ll[8]  + hl[2] + lh[2]
// // bit9   : ll[9]  + hl[3] + lh[3]
// // bit10  : ll[10] + hl[4] + lh[4]
// // bit11  : ll[11] + hl[5] + lh[5]
// // bit12  : hh[0]  + hl[6] + lh[6]
// // bit13  : hh[1]  + hl[7] + lh[7]
// // bit14  : hh[2] + (hh[4] & hh[3])
// // bit15  : carry from bit14
// // ------------------------------------------------------------

// wire cin9;
// wire [6:0] s_hi;
// wire [5:0] g_hi;
// wire g_top;
// wire [7:0] carry_hi;
// wire [7:0] sum_hi;

// assign prod[5:0] = ll[5:0];

// // ------------------------------------------------------------
// // Low direct prediction region: prod[6] ~ prod[8]
// // No CARRY4, no serial g dependency.
// // ------------------------------------------------------------

// // prod[6] = parity(col6)
// LUT6 #(
//     .INIT(64'h9696969696969696)
// ) LUT88_LOW6 (
//     .I0(ll[6]),
//     .I1(hl[0]),
//     .I2(lh[0]),
//     .I3(1'b0),
//     .I4(1'b0),
//     .I5(1'b0),
//     .O(prod[6])
// );

// // prod[7] = parity(col7) ^ majority(col6)
// LUT6 #(
//     .INIT(64'h6969699669969696)
// ) LUT88_LOW7 (
//     .I0(ll[7]),
//     .I1(hl[1]),
//     .I2(lh[1]),
//     .I3(ll[6]),
//     .I4(hl[0]),
//     .I5(lh[0]),
//     .O(prod[7])
// );

// // prod[8] = parity(col8) ^ majority(col7)
// LUT6 #(
//     .INIT(64'h6969699669969696)
// ) LUT88_LOW8 (
//     .I0(ll[8]),
//     .I1(hl[2]),
//     .I2(lh[2]),
//     .I3(ll[7]),
//     .I4(hl[1]),
//     .I5(lh[1]),
//     .O(prod[8])
// );

// // Predicted carry into high region bit9:
// // cin9 = majority(col8) | (parity(col8) & majority(col7))
// LUT6 #(
//     .INIT(64'hFEFEFEE8FEE8E8E8)
// ) LUT88_CIN9 (
//     .I0(ll[8]),
//     .I1(hl[2]),
//     .I2(lh[2]),
//     .I3(ll[7]),
//     .I4(hl[1]),
//     .I5(lh[1]),
//     .O(cin9)
// );

// // ------------------------------------------------------------
// // High PG + CARRY4 region: prod[9] ~ prod[15]
// // ------------------------------------------------------------

// // bit9 start: O5 = g9 = majority(col9), O6 = s9 = parity(col9)
// LUT6_2 #(
//     .INIT(64'h96969696E8E8E8E8)
// ) LUT88_PG9 (
//     .I0(ll[9]),
//     .I1(hl[3]),
//     .I2(lh[3]),
//     .I3(1'b0),
//     .I4(1'b0),
//     .I5(1'b1),
//     .O5(g_hi[0]),
//     .O6(s_hi[0])
// );

// // bit10: O5 = g10, O6 = s10 ^ g9
// LUT6_2 #(
//     .INIT(64'h69966996E8E8E8E8)
// ) LUT88_PG10 (
//     .I0(ll[10]),
//     .I1(hl[4]),
//     .I2(lh[4]),
//     .I3(g_hi[0]),
//     .I4(1'b0),
//     .I5(1'b1),
//     .O5(g_hi[1]),
//     .O6(s_hi[1])
// );

// // bit11: O5 = g11, O6 = s11 ^ g10
// LUT6_2 #(
//     .INIT(64'h69966996E8E8E8E8)
// ) LUT88_PG11 (
//     .I0(ll[11]),
//     .I1(hl[5]),
//     .I2(lh[5]),
//     .I3(g_hi[1]),
//     .I4(1'b0),
//     .I5(1'b1),
//     .O5(g_hi[2]),
//     .O6(s_hi[2])
// );

// // bit12: O5 = g12, O6 = s12 ^ g11
// LUT6_2 #(
//     .INIT(64'h69966996E8E8E8E8)
// ) LUT88_PG12 (
//     .I0(hh[0]),
//     .I1(hl[6]),
//     .I2(lh[6]),
//     .I3(g_hi[2]),
//     .I4(1'b0),
//     .I5(1'b1),
//     .O5(g_hi[3]),
//     .O6(s_hi[3])
// );

// // bit13: O5 = g13, O6 = s13 ^ g12
// LUT6_2 #(
//     .INIT(64'h69966996E8E8E8E8)
// ) LUT88_PG13 (
//     .I0(hh[1]),
//     .I1(hl[7]),
//     .I2(lh[7]),
//     .I3(g_hi[3]),
//     .I4(1'b0),
//     .I5(1'b1),
//     .O5(g_hi[4]),
//     .O6(s_hi[4])
// );

// // bit14 top of exact 2x2 high product:
// // top operand = hh[2] + (hh[4] & hh[3])
// // O5 = g_top = hh[2] & hh[4] & hh[3]
// // O6 = s14 ^ g13 = hh[2] ^ (hh[4] & hh[3]) ^ g13
// LUT6_2 #(
//     .INIT(64'h96669666A000A000)
// ) LUT88_TOP14 (
//     .I0(hh[2]),
//     .I1(g_hi[4]),
//     .I2(hh[4]),
//     .I3(hh[3]),
//     .I4(1'b0),
//     .I5(1'b1),
//     .O5(g_top),
//     .O6(s_hi[5])
// );

// // bit15 = g_top plus carry from bit14
// LUT6 #(
//     .INIT(64'hAAAAAAAAAAAAAAAA)
// ) LUT88_S15 (
//     .I0(g_top),
//     .I1(1'b0),
//     .I2(1'b0),
//     .I3(1'b0),
//     .I4(1'b0),
//     .I5(1'b0),
//     .O(s_hi[6])
// );

// // CARRY4 for prod[9] ~ prod[12]
// CARRY4 CARRY88_0 (
//     .CO     (carry_hi[3:0]),
//     .O      (sum_hi[3:0]),
//     .CI     (1'b0),
//     .CYINIT (cin9),
//     .DI     ({g_hi[2], g_hi[1], g_hi[0], 1'b0}),
//     .S      ({s_hi[3], s_hi[2], s_hi[1], s_hi[0]})
// );

// // CARRY4 for prod[13] ~ prod[15], upper one output ignored
// CARRY4 CARRY88_1 (
//     .CO     (carry_hi[7:4]),
//     .O      (sum_hi[7:4]),
//     .CI     (carry_hi[3]),
//     .CYINIT (1'b0),
//     .DI     ({1'b0, 1'b0, g_hi[4], g_hi[3]}),
//     .S      ({1'b0, s_hi[6], s_hi[5], s_hi[4]})
// );

// assign prod[12:9]  = sum_hi[3:0];
// assign prod[15:13] = sum_hi[6:4];

// endmodule