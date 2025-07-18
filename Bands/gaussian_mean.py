import numpy as np
from scipy.stats import rankdata
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm
import plotly.express as px

colors = px.colors.qualitative.Plotly[:3]

def simulate_F(n):
    mus = np.random.randn(n) * np.sqrt(0.5)
    return norm.cdf(x[None, :], loc=mus[:, None], scale=1.0)

# Parameters
np.random.seed(0)
B, N, alpha = 500_000, 500_000, 0.05
x = np.linspace(-3, 3, 100)

# --- 1) Generate B empirical CDFs from uniform samples ---
F_B = simulate_F(B)

# --- 2) Compute confidence bands ---
bands = {}

# Method A: quantile-based
bands['A'] = {
    'lower': np.quantile(F_B, alpha/2, axis=0),
    'upper': np.quantile(F_B, 1 - alpha/2, axis=0)
}

# Method B: t-statistic
mean_F = F_B.mean(axis=0)
std_F  = F_B.std(axis=0) + 1e-8
T      = (F_B - mean_F) / std_F
infT, supT = T.min(axis=1), T.max(axis=1)
q_low, q_up = np.quantile(infT, alpha/2), np.quantile(supT, 1 - alpha/2)
bands['B'] = {
    'lower': np.clip(mean_F + q_low * std_F, 0, 1),
    'upper': np.clip(mean_F + q_up  * std_F, 0, 1)
}

# Method C: rank-based quantile matching
def inv_ordinal(z, F):
    sorted_F = np.sort(F, axis=0)
    idx = int(np.clip(round(z), 0, F.shape[0] - 1))
    return sorted_F[idx]

ranks_max = np.vstack([rankdata(F_B[:, j], method='max')-1 for j in range(F_B.shape[1])]).T
ranks_min = np.vstack([rankdata(F_B[:, j], method='min')-1 for j in range(F_B.shape[1])]).T
infZ, supZ = ranks_max.min(axis=1), ranks_min.max(axis=1)
l0, u0 = np.quantile(infZ, alpha/2), np.quantile(supZ, 1 - alpha/2)
bands['C'] = {
    'lower': np.clip(inv_ordinal(l0, F_B), 0, 1),
    'upper': np.clip(inv_ordinal(u0, F_B), 0, 1)
}

# --- 3) Coverage check ---

F_new = simulate_F(N)
coverage = {m: np.mean(np.all((F_new >= bands[m]['lower']) & (F_new <= bands[m]['upper']), axis=1))
            for m in bands}
print("Coverage rates:")
for m, rate in coverage.items():
    print(f" Method {m}: {rate:.3f}")

# --- 4) Plotting with Plotly ---
labels = {'A': 'Pointwise', 'B': 'Student', 'C': 'Uniform'}

fig1 = make_subplots(
    rows=1, cols=3, shared_yaxes=True,
    subplot_titles=[f"{labels[m]} (Coverage={coverage[m]:.3f})" for m in ['A','B','C']]
)

subset_indices = np.random.choice(B, size=20, replace=False)
subset_cdfs = F_B[subset_indices]

for i, m in enumerate(['A', 'B', 'C']):
    for j in range(len(subset_cdfs)):
        fig1.add_trace(go.Scatter(
            x=x, y=subset_cdfs[j],
            mode='lines',
            line=dict(color='gray', width=1),
            showlegend=False
        ), row=1, col=i+1)

    fig1.add_trace(go.Scatter(
        x=x, y=mean_F,
        mode='lines', name='Mean F(x)',
        line=dict(color='black')
    ), row=1, col=i+1)

    fig1.add_trace(go.Scatter(
        x=x, y=bands[m]['upper'],
        mode='lines',
        line=dict(color=colors[i], width=1),
        showlegend=False
    ), row=1, col=i+1)

    fig1.add_trace(go.Scatter(
        x=x, y=bands[m]['lower'],
        mode='lines',
        line=dict(color=colors[i], width=1),
        fill='tonexty',
        name=f"{labels[m]} CI"
    ), row=1, col=i+1)

fig1.update_layout(
    title='Confidence Bands (3 Methods)',
    height=400,
    width=1000,
    showlegend=False,
    yaxis=dict(range=[-0.05, 1.05]),
    shapes=[
        dict(
            type='rect', xref='paper', yref='paper',
            x0=-0.04, x1=1/3 - 0.02, y0=-0.1, y1=1.15,
            line=dict(color='red', width=3)
        ),
        dict(
            type='rect', xref='paper', yref='paper',
            x0=1/3, x1=1.02, y0=-0.1, y1=1.15,
            line=dict(color='green', width=3)
        )
    ]
)

fig1.write_image("Bands_Gaussian_Mean.png", scale=2.0)

# Width plot
fig2 = go.Figure()
for i, m in enumerate(bands):
    length = bands[m]['upper'] - bands[m]['lower']
    fig2.add_trace(go.Scatter(x=x, y=length, mode='lines', line=dict(color=colors[i]), name=f"{labels[m]} (Mean Width={length.mean():.3f})"))
fig2.update_layout(
    title='Pointwise Width',
    xaxis_title='x',
    yaxis_title='Width',
    legend=dict(font=dict(size=14))
)

fig2.write_image("Width_Gaussian_Mean.png", scale=2.0)
