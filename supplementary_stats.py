"""
Supplementary statistical tests for Rodriguez (2026)
Bose-Einstein Statistics of the Radial Acceleration Relation

Run from same directory as rar_bose_einstein.py after generating
bose_einstein_plotdata.csv.

USAGE:  python3 supplementary_stats.py
"""

import numpy as np
from scipy import stats
import json

# ============================================================
# LOAD DATA
# ============================================================
data = np.genfromtxt("bose_einstein_plotdata.csv", delimiter=",",
                     skip_header=1)

theta    = data[:, 0]
D_pred   = data[:, 3]
D_obs    = data[:, 4]
gbar_log = data[:, 5]

N = len(theta)
resid = np.log10(D_obs) - np.log10(D_pred)

print(f"N = {N}")
print(f"Residual mean  = {np.mean(resid):+.4f} dex")
print(f"Residual std   = {np.std(resid):.4f} dex")
print()

results = {}

# ============================================================
# 1. SPEARMAN RANK CORRELATION: residuals vs log(g_bar)
#    Tests for any monotonic trend in residuals across
#    the acceleration range.
# ============================================================
rho_gbar, p_gbar = stats.spearmanr(gbar_log, resid)
print("=" * 60)
print("1. SPEARMAN RANK CORRELATION: residuals vs log(g_bar)")
print("=" * 60)
print(f"   rho = {rho_gbar:+.4f}")
print(f"   p   = {p_gbar:.2e}")
if p_gbar > 0.05:
    print(f"   --> No significant monotonic trend (p > 0.05)")
else:
    print(f"   --> Significant trend detected (p < 0.05)")
print()

results["spearman_gbar"] = {
    "rho": float(rho_gbar),
    "p": float(p_gbar)
}

# ============================================================
# 2. SPEARMAN RANK CORRELATION: residuals vs theta
#    Same test in the natural variable of the model.
# ============================================================
rho_theta, p_theta = stats.spearmanr(theta, resid)
print("=" * 60)
print("2. SPEARMAN RANK CORRELATION: residuals vs theta")
print("=" * 60)
print(f"   rho = {rho_theta:+.4f}")
print(f"   p   = {p_theta:.2e}")
if p_theta > 0.05:
    print(f"   --> No significant monotonic trend (p > 0.05)")
else:
    print(f"   --> Significant trend detected (p < 0.05)")
print()

results["spearman_theta"] = {
    "rho": float(rho_theta),
    "p": float(p_theta)
}

# ============================================================
# 3. ANDERSON-DARLING TEST: normality of residuals
#    Tests whether residuals are normally distributed,
#    as expected if scatter is dominated by Gaussian
#    measurement error.
# ============================================================
ad_stat, ad_crit, ad_sig = stats.anderson(resid, dist='norm')
print("=" * 60)
print("3. ANDERSON-DARLING TEST: normality of residuals")
print("=" * 60)
print(f"   Statistic = {ad_stat:.4f}")
print(f"   Critical values and significance levels:")
for cv, sl in zip(ad_crit, ad_sig):
    flag = "REJECT" if ad_stat > cv else "do not reject"
    print(f"     {sl:5.1f}%: critical = {cv:.4f}  --> {flag}")
print()

results["anderson_darling"] = {
    "statistic": float(ad_stat),
    "critical_5pct": float(ad_crit[2]),
    "normal": bool(ad_stat < ad_crit[2])
}

# ============================================================
# 4. REDUCED CHI-SQUARED
#    Using observational errors propagated to log10(D).
#    If errors are not available, use intrinsic scatter
#    sigma = 0.13 dex as uniform error estimate.
# ============================================================
sigma_intrinsic = 0.13  # dex, from Lelli et al. 2017
chi2 = np.sum((resid / sigma_intrinsic)**2)
chi2_red = chi2 / (N - 1)  # 1 parameter: a0 (held fixed, but conventional)
print("=" * 60)
print("4. REDUCED CHI-SQUARED")
print("=" * 60)
print(f"   sigma assumed   = {sigma_intrinsic} dex (Lelli et al. 2017)")
print(f"   chi2            = {chi2:.1f}")
print(f"   chi2 / (N-1)    = {chi2_red:.4f}")
print(f"   Expected: ~1.0 if model matches data within errors")
if 0.8 < chi2_red < 1.2:
    print(f"   --> GOOD FIT")
elif chi2_red < 0.8:
    print(f"   --> Errors may be overestimated")
else:
    print(f"   --> Possible underestimated errors or model tension")
print()

results["reduced_chi2"] = {
    "chi2": float(chi2),
    "dof": int(N - 1),
    "chi2_reduced": float(chi2_red),
    "sigma_assumed_dex": sigma_intrinsic
}

# ============================================================
# 5. MEDIAN ABSOLUTE DEVIATION (robust scatter estimate)
#    Less sensitive to outliers than std.
# ============================================================
mad = np.median(np.abs(resid - np.median(resid)))
# For normal distribution: sigma ~ 1.4826 * MAD
sigma_mad = 1.4826 * mad
print("=" * 60)
print("5. MEDIAN ABSOLUTE DEVIATION")
print("=" * 60)
print(f"   MAD             = {mad:.4f} dex")
print(f"   sigma (MAD)     = {sigma_mad:.4f} dex")
print(f"   sigma (std)     = {np.std(resid):.4f} dex")
print(f"   Ratio std/MAD   = {np.std(resid)/sigma_mad:.3f}")
print(f"   Expected ratio for Gaussian: 1.000")
print()

results["mad"] = {
    "mad": float(mad),
    "sigma_mad": float(sigma_mad),
    "sigma_std": float(np.std(resid))
}

# ============================================================
# SAVE
# ============================================================
with open("supplementary_stats.json", "w") as f:
    json.dump(results, f, indent=2)

print("=" * 60)
print("Results saved: supplementary_stats.json")
print("=" * 60)
