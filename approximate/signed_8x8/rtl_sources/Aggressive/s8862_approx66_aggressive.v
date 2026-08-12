// Aggressive LUT-only approximation of the unsigned AL*BL block.
//
// This is a renamed, self-contained copy of the best local
// "unshared_paircomp" candidate.  All three 6x2 truth tables and the
// pairwise 6x6 compressor are approximate.  The surrounding signed fused
// accumulator stages are still exact.
//
// Resources in this core: 14 LUT6_2 + 4 LUT6, no CARRY4.
module s8862_approx66_aggressive_core (
    input  wire [5:0] a,
    input  wire [5:0] b,
    output wire [11:0] prod
);

wire [7:0] plow;
wire [7:0] pmid;
wire [7:0] phigh;

s8862_aggr62_low  low_block  (.a(a), .b(b[1:0]), .prod(plow));
s8862_aggr62_mid  mid_block  (.a(a), .b(b[3:2]), .prod(pmid));
s8862_aggr62_high high_block (.a(a), .b(b[5:4]), .prod(phigh));

s8862_aggr_comp66 compressor (
    .plow(plow), .pmid(pmid), .phigh(phigh), .prod(prod)
);

endmodule


module s8862_aggr62_low (
    input wire [5:0] a, input wire [1:0] b, output wire [7:0] prod
);

LUT6_2 #(.INIT(64'heac0d38ba0a02517)) lut01 (
    .I0(b[0]), .I1(b[1]), .I2(a[0]), .I3(a[1]),
    .I4(1'b1), .I5(1'b1), .O5(prod[0]), .O6(prod[1])
);
LUT6_2 #(.INIT(64'heeaacc00eac0eac0)) lut23 (
    .I0(b[0]), .I1(b[1]), .I2(a[1]), .I3(a[2]),
    .I4(a[3]), .I5(1'b1), .O5(prod[2]), .O6(prod[3])
);
LUT6_2 #(.INIT(64'hee22cc00ea40eac0)) lut45 (
    .I0(b[0]), .I1(b[1]), .I2(a[3]), .I3(a[4]),
    .I4(a[5]), .I5(1'b1), .O5(prod[4]), .O6(prod[5])
);
LUT6_2 #(.INIT(64'hc80039b3c4001018)) lut67 (
    .I0(b[0]), .I1(b[1]), .I2(a[4]), .I3(a[5]),
    .I4(1'b1), .I5(1'b1), .O5(prod[6]), .O6(prod[7])
);

endmodule


module s8862_aggr62_mid (
    input wire [5:0] a, input wire [1:0] b, output wire [7:0] prod
);

LUT6_2 #(.INIT(64'heac0d38ba0a02517)) lut01 (
    .I0(b[0]), .I1(b[1]), .I2(a[0]), .I3(a[1]),
    .I4(1'b1), .I5(1'b1), .O5(prod[0]), .O6(prod[1])
);
LUT6_2 #(.INIT(64'hee22cc00e240eac0)) lut23 (
    .I0(b[0]), .I1(b[1]), .I2(a[1]), .I3(a[2]),
    .I4(a[3]), .I5(1'b1), .O5(prod[2]), .O6(prod[3])
);
LUT6_2 #(.INIT(64'hacaa4c80a8c0ea40)) lut45 (
    .I0(b[0]), .I1(b[1]), .I2(a[3]), .I3(a[4]),
    .I4(a[5]), .I5(1'b1), .O5(prod[4]), .O6(prod[5])
);
LUT6_2 #(.INIT(64'hc80039b32c801018)) lut67 (
    .I0(b[0]), .I1(b[1]), .I2(a[4]), .I3(a[5]),
    .I4(1'b1), .I5(1'b1), .O5(prod[6]), .O6(prod[7])
);

endmodule


module s8862_aggr62_high (
    input wire [5:0] a, input wire [1:0] b, output wire [7:0] prod
);

LUT6_2 #(.INIT(64'heac0d38ba0a02517)) lut01 (
    .I0(b[0]), .I1(b[1]), .I2(a[0]), .I3(a[1]),
    .I4(1'b1), .I5(1'b1), .O5(prod[0]), .O6(prod[1])
);
LUT6_2 #(.INIT(64'hee22cc00e240eac0)) lut23 (
    .I0(b[0]), .I1(b[1]), .I2(a[1]), .I3(a[2]),
    .I4(a[3]), .I5(1'b1), .O5(prod[2]), .O6(prod[3])
);
LUT6_2 #(.INIT(64'hf02a4d898862e6c0)) lut45 (
    .I0(b[0]), .I1(b[1]), .I2(a[3]), .I3(a[4]),
    .I4(a[5]), .I5(1'b1), .O5(prod[4]), .O6(prod[5])
);
LUT6_2 #(.INIT(64'hc80039b324801018)) lut67 (
    .I0(b[0]), .I1(b[1]), .I2(a[4]), .I3(a[5]),
    .I4(1'b1), .I5(1'b1), .O5(prod[6]), .O6(prod[7])
);

endmodule


module s8862_aggr_comp66 (
    input  wire [7:0] plow,
    input  wire [7:0] pmid,
    input  wire [7:0] phigh,
    output wire [11:0] prod
);

assign prod[0]  = plow[0];
assign prod[1]  = plow[1];
assign prod[10] = phigh[6];
assign prod[11] = phigh[7];

LUT6_2 #(.INIT(64'h42f1fff85d55fee6)) comp23_lut (
    .I0(plow[2]), .I1(pmid[0]), .I2(plow[3]), .I3(pmid[1]),
    .I4(1'b0), .I5(1'b1), .O5(prod[2]), .O6(prod[3])
);
LUT6 #(.INIT(64'hfffffffefffefe96)) comp4_lut (
    .I0(plow[4]), .I1(pmid[2]), .I2(phigh[0]),
    .I3(plow[5]), .I4(pmid[3]), .I5(phigh[1]), .O(prod[4])
);
LUT6 #(.INIT(64'hffffffffffffffe8)) comp5_lut (
    .I0(plow[4]), .I1(pmid[2]), .I2(phigh[0]),
    .I3(plow[5]), .I4(pmid[3]), .I5(phigh[1]), .O(prod[5])
);
LUT6 #(.INIT(64'hfdff75feff7e5696)) comp6_lut (
    .I0(plow[6]), .I1(pmid[4]), .I2(phigh[2]),
    .I3(plow[7]), .I4(pmid[5]), .I5(phigh[3]), .O(prod[6])
);
LUT6 #(.INIT(64'hfdffddffdfbf55e8)) comp7_lut (
    .I0(plow[6]), .I1(pmid[4]), .I2(phigh[2]),
    .I3(plow[7]), .I4(pmid[5]), .I5(phigh[3]), .O(prod[7])
);
LUT6_2 #(.INIT(64'hc20bdcc81222c8f6)) comp89_lut (
    .I0(pmid[6]), .I1(phigh[4]), .I2(pmid[7]), .I3(phigh[5]),
    .I4(1'b0), .I5(1'b1), .O5(prod[8]), .O6(prod[9])
);

endmodule


// Complete aggressive signed 8x8 top.
// Expected explicit resources: 31 LUT6_2 + 4 LUT6 + 4 CARRY4.
(* use_dsp = "no" *)
module signed88_approx_aggressive (
    input  wire signed [7:0]  a,
    input  wire signed [7:0]  b,
    output wire signed [15:0] prod
);

wire [11:0] ll;
wire stage1_q6;
wire q6_unused;

s8862_approx66_aggressive_core low_multiplier (
    .a(a[5:0]), .b(b[5:0]), .prod(ll)
);

// The LUT-only approximate compressor has no compatible spare output, so q6
// costs one standalone LUT in this most aggressive configuration.
LUT6_2 #(
    .INIT(64'h00000000d0d0d0d0)
) stage1_q6_lut (
    .I0(b[5]), .I1(a[6]), .I2(a[7]),
    .I3(1'b1), .I4(1'b1), .I5(1'b1),
    .O5(stage1_q6), .O6(q6_unused)
);

s8862_signed_finish finish (
    .a(a), .b(b), .ll(ll), .stage1_q6(stage1_q6), .prod(prod)
);

endmodule
