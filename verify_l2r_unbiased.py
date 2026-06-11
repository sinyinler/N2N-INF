"""数值验证：把 L2R 重腐蚀自监督融进多帧（网络额外看到邻帧）后，无偏性是否还成立。

L2R 目标(α=1)：L = (f(y₁)-y)² + 2·f(y₁)·h，  y₁=y+h，h=σw（高斯重腐蚀器）。
对 f 求极小 ⟺ 回归到伪目标 t = y - h。
"无偏"= L2R 最优 f 与有监督最优 f(回归真值 x)一致 ⟹ 拿到的是 MMSE 最优去噪器。

对照三种输入：
  A 只看 y₁（标准单帧 L2R）
  B 看 y₁ + 独立邻帧均值（融多帧，噪声独立）
  C 看 y₁ + 与中心噪声相关的邻帧（违反独立条件）
"""
import numpy as np

rng = np.random.default_rng(0)
N = 4_000_000
sigma = 1.0          # 噪声 std
M = 4                # 邻帧数
rho = 0.7            # C 情形：邻帧噪声与中心噪声相关系数

x = rng.uniform(-3.0, 3.0, N)                  # 干净信号，Var≈3
n = rng.normal(0, sigma, N)                    # 中心帧噪声
w = rng.normal(0, 1, N)                         # 重腐蚀随机源
h = sigma * w                                   # 高斯重腐蚀器 h(w)=σw
y = x + n
y1 = y + h                                      # y₁ = y + α·h, α=1
t = y - h                                       # L2R 伪目标 (= argmin 的解)

# 邻帧均值：独立 / 相关
e_indep = rng.normal(0, sigma, (M, N)).mean(0)                       # Var=σ²/M, ⊥ n
mbar_indep = x + e_indep
e_corr_each = rho * n + np.sqrt(1 - rho**2) * rng.normal(0, sigma, (M, N))  # Cov(e,n)=ρσ²
mbar_corr = x + e_corr_each.mean(0)


def fit_predict(features, target):
    """最小二乘线性拟合 target~features，返回 (系数, 预测)。features 不含常数项。"""
    A = np.column_stack([np.ones(N)] + features)
    coef, *_ = np.linalg.lstsq(A, target, rcond=None)
    return coef, A @ coef


def report(name, feats):
    # L2R：回归伪目标 t；有监督 oracle：回归真值 x
    c_l2r, pred_l2r = fit_predict(feats, t)
    c_sup, pred_sup = fit_predict(feats, x)
    # 朴素(漏掉 2<f,h> 校正项)：直接回归 y，会过平滑/有偏
    c_naive, pred_naive = fit_predict(feats, y)
    bias = pred_l2r.mean() - x.mean()
    mse_l2r = np.mean((pred_l2r - x) ** 2)
    mse_sup = np.mean((pred_sup - x) ** 2)
    mse_naive = np.mean((pred_naive - x) ** 2)
    coef_match = np.max(np.abs(c_l2r - c_sup))
    print(f"\n[{name}]  特征={['y1']+['mbar'] * (len(feats) - 1)}")
    print(f"  L2R系数   {np.round(c_l2r,4)}")
    print(f"  有监督系数 {np.round(c_sup,4)}   |Δ系数|max={coef_match:.5f}  -> {'一致(无偏✓)' if coef_match<1e-2 else '不一致(有偏✗)'}")
    print(f"  朴素(无校正)系数 {np.round(c_naive,4)}  -> {'明显不同(校正项必要)' if np.max(np.abs(c_naive-c_sup))>1e-2 else '同'}")
    print(f"  整体偏差 mean(f)-mean(x) = {bias:+.5f}")
    print(f"  MSE→clean:  L2R={mse_l2r:.4f}  有监督最优={mse_sup:.4f}  朴素={mse_naive:.4f}")
    # 分信号档位看偏差（确认不是平均抵消）
    edges = np.linspace(-3, 3, 7)
    binbias = [ (pred_l2r[(x>=edges[i])&(x<edges[i+1])] - x[(x>=edges[i])&(x<edges[i+1])]).mean()
                for i in range(6) ]
    print(f"  分档偏差(6档): {np.round(binbias,4)}")
    return mse_l2r


print("=" * 70)
print(f"N={N}, σ={sigma}, 邻帧数 M={M}")
mse_A = report("A 单帧 L2R", [y1])
mse_B = report("B 融多帧·独立邻帧", [y1, mbar_indep])
mse_C = report("C 融多帧·相关邻帧(ρ=0.7, 违反独立)", [y1, mbar_corr])

print("\n" + "=" * 70)
print("结论：")
print(f"  A,B 的 L2R系数 == 有监督系数 → 无偏性保持；C 不等 → 相关邻帧引入偏差。")
print(f"  多帧降噪收益：MSE  A(单帧)={mse_A:.4f}  ->  B(独立多帧)={mse_B:.4f}  "
      f"(降 {100*(1-mse_B/mse_A):.1f}%)")
