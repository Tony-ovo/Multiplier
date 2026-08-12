`timescale 1ns / 1ps
module tb_approx66;

reg  [5:0] a;      // 6-bit 输入
reg  [5:0] b;      // 6-bit 输入
wire [11:0] prod;  // 12-bit 输出 (6+6=12)

// 请确保这里的模块名与你的6x6乘法器设计模块名一致
approx66 uut (
    .a(a),
    .b(b),
    .prod(prod)
);

integer i, j;
reg [11:0] golden; // 最大值为 63*63 = 3969，12位即可装下

// ===== 参数 =====
parameter TOTAL = 4096;    // 2^6 * 2^6 = 64 * 64 = 4096
parameter D = 3969;        // (2^6 - 1) * (2^6 - 1) = 63 * 63 = 3969

// ===== 统计变量 =====
integer total_cnt;
integer error_cnt;

integer ED;
integer ED_sum;
integer WCE;

real RED_sum;   // Σ(ED / M)
real MED;
real NED;
real ER;
real MRED;

initial begin
    $display("====== Approximate 6x6 Multiplier Test (Paper Definition) ======");

    total_cnt = 0;
    error_cnt = 0;

    ED_sum = 0;
    RED_sum = 0.0;
    WCE = 0;

    // a 的范围是 0 到 63 (2^6 - 1)
    for (i = 0; i < 64; i = i + 1) begin
        // b 的范围是 0 到 63 (2^6 - 1)
        for (j = 0; j < 64; j = j + 1) begin
            
            a = i;
            b = j;

            #1;

            golden = a * b;
            total_cnt = total_cnt + 1;

            // ===== ED (Error Distance) =====
            ED = (prod > golden) ? (prod - golden) : (golden - prod);
            ED_sum = ED_sum + ED;

            // ===== WCE (Worst-Case Error) =====
            if (ED > WCE)
                WCE = ED;

            // ===== ER (Error Rate) =====
            if (ED != 0)
                error_cnt = error_cnt + 1;

            // ===== RED（论文定义）=====
            if (golden != 0)
                RED_sum = RED_sum + (ED * 1.0 / golden);

        end
    end

    // ===== 指标计算（严格论文）=====

    // ER
    ER = error_cnt * 1.0 / total_cnt;

    // MED
    MED = ED_sum * 1.0 / TOTAL;

    // NED = MED / D
    NED = MED / D;

    // MRED = RED / TOTAL
    MRED = RED_sum / TOTAL;

    // ===== 输出 =====
    $display("\n====== Metrics (Paper Accurate Definition) ======");
    $display("Total Cases      = %d", TOTAL);
    $display("Error Cases      = %d", error_cnt);

    $display("ER               = %f", ER);
    $display("MED              = %f", MED);
    $display("NED              = %f", NED);
    $display("MRED             = %f", MRED);
    $display("WCE              = %d", WCE);

    $display("====== Test Finished ======");
    $finish;
end

endmodule
