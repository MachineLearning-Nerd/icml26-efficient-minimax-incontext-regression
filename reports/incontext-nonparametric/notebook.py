"""Minimax-optimal in-context nonparametric regression with transformers — a tour.

A self-contained marimo notebook that opens with the reproduction's headline
evidence and walks through the paper's central claim and how each part was
verified. Figures are fetched from the public GitHub raw URLs (the repo is
public), so the notebook renders in Molab without re-running experiments.

Run locally:  marimo edit reports/incontext-nonparametric/notebook.py
Run headless: marimo run  reports/incontext-nonparametric/notebook.py
"""
import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium", css_file="")

FIG = ("https://raw.githubusercontent.com/MachineLearning-Nerd/"
       "icml26-efficient-minimax-incontext-regression/main/reports/"
       "incontext-nonparametric/images")


@app.cell
def _(mo):
    mo.md(
        r"""
        # Minimax-optimal in-context nonparametric regression with transformers

        **arXiv 2601.15014 (ICML 2026).** Can a transformer do nonparametric
        regression *as well as the optimal estimator* — using only
        **$\Theta(\log n)$ parameters**?

        The paper says yes: for $\alpha$-Hölder regression functions in $d$
        dimensions, a transformer pretrained on
        $\Gamma \geq n^{2\alpha/(2\alpha+d)}\log^3(en)$ sequences achieves the
        minimax-optimal rate $n^{-2\alpha/(2\alpha+d)}$. Its mechanism: build a
        kernel-weighted monomial basis with ReLU feed-forward nets, then run
        gradient descent in linear attention to solve the local-polynomial
        least-squares problem.

        This notebook opens with the reproduction's headline result, then walks
        through how every scoped claim was verified. **6/6 scoped claim
        contracts pass** (5 HIGH, 1 MEDIUM). Paper-level status is
        **INCONCLUSIVE**: the global ERM, cited lower bounds, and GPU training
        dynamics are outside this audit.
        """
    )
    return


@app.cell
def _(mo):
    mo.md("## The headline result — the local-polynomial minimax rate")
    return


@app.cell
def _(mo):
    # Headline figure: excess risk at the Hölder cusp follows n^{-2a/(2a+d)}.
    mo.image(src=f"{FIG}/fig1_locpol_rate.png")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        Pointwise excess risk of the local polynomial estimator (degree
        $p=\lceil\alpha\rceil$, kernel $K=(1-\|x\|_1)_+^2$, bandwidth
        $h=n^{-1/(2\alpha+d)}$) at the cusp of an exactly-$\alpha$-Hölder
        function $m(x)=\|x-x_0\|^\alpha$. Measured log-log slopes match the
        theoretical rate $-2\alpha/(2\alpha+d)$ (e.g. $-0.76$ vs $-0.75$ at
        $\alpha{=}1.5, d{=}1$). Dashed guides are the theoretical rates.
        """
    )
    return


@app.cell
def _(mo, n_slider, alpha_slider, d_slider, rate):
    mo.md(
        rf"""
        ### The rate, interactively

        The minimax rate for $\alpha$-Hölder functions in $d$ dimensions with
        $n$ context examples is $n^{{-2\alpha/(2\alpha+d)}}$.

        With $\alpha={alpha_slider.value}$, $d={d_slider.value}$,
        $n={n_slider.value}$: the rate is
        $\mathbf{{{rate:.4f}}}$, and the prescribed bandwidth is
        $h = n^{{-1/(2\alpha+d)}} = {n_slider.value ** (-1/(2*alpha_slider.value + d_slider.value)):.4f}$.
        """
    )
    return


@app.cell
def _(mo, rate):
    rate  # noqa
    return (rate,)


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    alpha_slider = mo.slider(0.5, 4.0, step=0.5, value=1.5, label=r"smoothness $\alpha$")
    d_slider = mo.slider(1, 4, step=1, value=1, label=r"dimension $d$")
    n_slider = mo.slider(32, 4096, step=32, value=512, label=r"context $n$")
    return alpha_slider, d_slider, n_slider


@app.cell
def _(alpha_slider, d_slider, n_slider):
    rate = n_slider.value ** (-2 * alpha_slider.value / (2 * alpha_slider.value + d_slider.value))
    return (rate,)


@app.cell
def _(mo):
    mo.md(
        r"""
        ## The construction: a transformer that *is* the estimator

        The paper's existence theorem (Theorem 3.1) builds a concrete transformer
        $f_{\mathrm{TF}}$ that approximates local polynomial estimation to
        $O(1/n)$. Three verified mechanisms (Section 4):

        1. **ReLU FFN builds the kernel — exactly.** $K=(1-\|x\|_1)_+^2$ is
           piecewise linear, so its square root $(1-\|u\|_1)_+$ is an *exact*
           2-hidden-layer ReLU network (max error 0).
        2. **ReLU FFN builds the monomial basis** $u^\nu$ to arbitrary precision.
        3. **Linear attention runs gradient descent.** One attention block
           computes the sufficient statistics $\sum a_i a_i^\top$, $\sum a_i y_i$
           *exactly*; $\Theta(\log n)$ blocks converge to the least-squares
           solution (the normal matrix has constant condition number, by
           Theorem 2.5's eigenvalue bound).
        """
    )
    return


@app.cell
def _(mo):
    mo.image(src=f"{FIG}/fig3_construction_components.png")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ### The approximation error is $O(1/n)$

        Built from the verified construction, the pointwise RMS error
        $\|f_{\mathrm{TF}}-f_{\mathrm{LocPol}}\|_2$ has log-log slope $-2.43$ in
        $n$. Since both are bounded by $M$,
        $|R(f_{\mathrm{TF}})-R(f_{\mathrm{LocPol}})| \le 4M\|f_{\mathrm{TF}}-f_{\mathrm{LocPol}}\|_2 = O(1/n)$.
        """
    )
    return


@app.cell
def _(mo):
    mo.image(src=f"{FIG}/fig2_construction_approx.png")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Parameter efficiency: $\Theta(\log n)$, not $\Theta(n)$

        Counting scalar parameters from Definitions 2.1–2.4: total =
        $L \times \text{per-block}$, with $L=\lceil C\log(en)\rceil$ and the
        per-block count depending only on $(d,\alpha)$ — so the total is
        $\Theta(\log n)$. (The bound $B=Cn^2$ is the *per-entry magnitude*
        bound, not a count; the prior revision misread it.)

        By **growth rate**, our parameter count grows with log-log slope
        $\approx 0.08$, far below Shen's $\Theta(n)$ (slope 1) and Kim's
        $\Theta(n^{d/(2\alpha+d)})$.
        """
    )
    return


@app.cell
def _(mo):
    mo.image(src=f"{FIG}/fig4_param_count.png")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Verdicts

        | Claim | Statement | Status | Confidence |
        |---|---|---|---|
        | C1 | locpol rate $n^{-2\alpha/(2\alpha+d)}$ | **VERIFIED** | HIGH |
        | C2 | transformer $\approx$ locpol, $O(1/n)$ | **VERIFIED** | HIGH |
        | C3 | ERM transformer minimax rate | **VERIFIED** | MEDIUM |
        | C4 | $L=\lceil C\log(en)\rceil$, $\Theta(\log n)$ params | **VERIFIED** | HIGH |
        | C5 | $\Gamma \geq n^{2\alpha/(2\alpha+d)}\log^3(en)$, smaller | **VERIFIED** | HIGH |
        | C6 | ReLU-FFN basis + linear-attention GD | **VERIFIED** | HIGH |

        Full evidence, raw CSV/JSON, negative controls, and limitations: see the
        [illustrated report](https://github.com/MachineLearning-Nerd/icml26-efficient-minimax-incontext-regression/blob/main/reports/incontext-nonparametric/report.md)
        and the claim verifiers under `repro/claims/`. Reproduce with
        `bash repro/ci.sh`.
        """
    )
    return


if __name__ == "__main__":
    app.run()
