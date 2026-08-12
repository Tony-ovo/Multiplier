//approx--comp66--all-OR

module comp66 (
    input wire [7:0] plow,
    input wire [7:0] pmid,
    input wire [7:0] phigh,

    output wire [11:0] prod
);

// Non-overlap bits.
assign prod[0]  = plow[0];
assign prod[1]  = plow[1];
assign prod[10] = phigh[6];
assign prod[11] = phigh[7];

// prod[2] = plow[2] | pmid[0]
// prod[3] = plow[3] | pmid[1]
LUT6_2 #(
    .INIT(64'h0000FFF80000FEE6)
) u_or23 (
    .I0(plow[2]),
    .I1(pmid[0]),
    .I2(plow[3]),
    .I3(pmid[1]),
    .I4(1'b0),
    .I5(1'b1),
    .O5(prod[2]),
    .O6(prod[3])
);

// 3-input OR LUT: output = I0 | I1 | I2. Unused inputs are tied to 0.
// prod[4] = plow[4] | pmid[2] | phigh[0]
LUT6 #(
    .INIT(64'h00000000000000FE)
) u_or4 (
    .I0(plow[4]),
    .I1(pmid[2]),
    .I2(phigh[0]),
    .I3(1'b0),
    .I4(1'b0),
    .I5(1'b0),
    .O(prod[4])
);

// prod[5] = plow[5] | pmid[3] | phigh[1]
LUT6 #(
    .INIT(64'h00000000000000FE)
) u_or5 (
    .I0(plow[5]),
    .I1(pmid[3]),
    .I2(phigh[1]),
    .I3(1'b0),
    .I4(1'b0),
    .I5(1'b0),
    .O(prod[5])
);

// prod[6] = plow[6] | pmid[4] | phigh[2]
LUT6 #(
    .INIT(64'h00000000000000FE)
) u_or6 (
    .I0(plow[6]),
    .I1(pmid[4]),
    .I2(phigh[2]),
    .I3(1'b0),
    .I4(1'b0),
    .I5(1'b0),
    .O(prod[6])
);

// prod[7] = plow[7] | pmid[5] | phigh[3]
LUT6 #(
    .INIT(64'h00000000000000FE)
) u_or7 (
    .I0(plow[7]),
    .I1(pmid[5]),
    .I2(phigh[3]),
    .I3(1'b0),
    .I4(1'b0),
    .I5(1'b0),
    .O(prod[7])
);

// prod[8] = pmid[6] | phigh[4]
// prod[9] = pmid[7] | phigh[5]
LUT6_2 #(
    .INIT(64'h00005F5800005E4E)
) u_or89 (
    .I0(pmid[6]),
    .I1(phigh[4]),
    .I2(pmid[7]),
    .I3(phigh[5]),
    .I4(1'b0),
    .I5(1'b1),
    .O5(prod[8]),
    .O6(prod[9])
);

endmodule

