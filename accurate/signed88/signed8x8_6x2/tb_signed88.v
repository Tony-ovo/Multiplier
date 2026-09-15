`timescale 1ns/1ps

module tb_signed88_6x2;

reg  signed [7:0]  a;
reg  signed [7:0]  b;
wire signed [15:0] prod;

reg signed [15:0] golden;

integer k;
integer errors;

signed88_6x2 dut (
    .a    (a),
    .b    (b),
    .prod (prod)
);

initial begin
    a      = 8'sd0;
    b      = 8'sd0;
    golden = 16'sd0;
    errors = 0;

    /*
     * k[15:8] 遍历 a 的所有 256 种二进制编码；
     * k[7:0]  遍历 b 的所有 256 种二进制编码。
     *
     * 因此仍然覆盖全部 256×256=65536 种输入。
     */
    for (k = 0; k < 65536; k = k + 1) begin
        a = k[15:8];
        b = k[7:0];

        // 等待组合逻辑稳定
        #1;

        golden = $signed(a) * $signed(b);

        if (prod !== golden) begin
            errors = errors + 1;

            // 最多打印前20个错误，避免输出过多
            if (errors <= 20) begin
                $display(
                    "ERROR: a=%0d, b=%0d, got=%0d (0x%04h), expected=%0d (0x%04h)",
                    $signed(a),
                    $signed(b),
                    $signed(prod),
                    prod,
                    $signed(golden),
                    golden
                );
            end
        end
    end

    if (errors == 0) begin
        $display("--------------------------------------------------");
        $display("PASS: all 65536 signed 8x8 combinations passed.");
        $display("--------------------------------------------------");
    end
    else begin
        $display("--------------------------------------------------");
        $display("FAIL: total mismatches = %0d", errors);
        $display("--------------------------------------------------");
    end

    $finish;
end

endmodule