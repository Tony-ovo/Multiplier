module accurate62 (
    input wire [5:0] a,
    input wire [1:0] b,

    output wire [7:0] prod
);

wire cin;
wire [3:0] gen,prop;


LUT6_2 #(
.INIT(64'h6ac06ac0a0a0a0a0)
) LUT6_inst1 (

.I0(b[0]), 
.I1(b[1]), 
.I2(a[0]), 
.I3(a[1]), 
.I4(1'd1), 
.I5(1'd1),

.O6(prod[1]),
.O5(prod[0])
);


LUT6_2 #(
.INIT(64'h88008000e6aa4c00)
) LUT6_inst2 (

.I0(b[0]), 
.I1(b[1]), 
.I2(a[0]), 
.I3(a[1]), 
.I4(a[2]), 
.I5(1'd1),

.O6(cin),
.O5(prod[2])
);


LUT6_2 #(
.INIT(64'h800080006ac06ac0)
) LUT6_inst3 (

.I0(b[0]), 
.I1(b[1]), 
.I2(a[2]), 
.I3(a[3]), 
.I4(1'd1), 
.I5(1'd1),

.O6(gen[0]),
.O5(prop[0])
);


LUT6_2 #(
.INIT(64'h800080006ac06ac0)
) LUT6_inst4 (

.I0(b[0]), 
.I1(b[1]), 
.I2(a[3]), 
.I3(a[4]), 
.I4(1'd1), 
.I5(1'd1),

.O6(gen[1]),
.O5(prop[1])
);


LUT6_2 #(
.INIT(64'h800080006ac06ac0)
) LUT6_inst5 (

.I0(b[0]), 
.I1(b[1]), 
.I2(a[4]), 
.I3(a[5]), 
.I4(1'd1), 
.I5(1'd1),

.O6(gen[2]),
.O5(prop[2])
);

LUT6_2 #(
.INIT(64'h0000000088888888)
) LUT6_inst6 (

.I0(b[1]), 
.I1(a[5]), 
.I2(1'd1), 
.I3(1'd1), 
.I4(1'd1), 
.I5(1'd1),

.O6(gen[3]),
.O5(prop[3])
);


//carry chain
wire [3:0] cout,sum;

CARRY4 CARRY4_inst (
.CO(cout), // 4-bit carry out
.O(sum), // 4-bit carry chain XOR data out
.CI(cin), // 1-bit carry cascade input
.CYINIT(1'd0), // 1-bit carry initialization
.DI(gen), // 4-bit carry-MUX data in
.S(prop) // 4-bit carry-MUX select input
);

assign prod[6:3] = sum[3:0];
assign prod[7] = cout[3];
 
endmodule