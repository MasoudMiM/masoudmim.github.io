---
layout: page
permalink: /uncertainty-calc/
title: Uncertainty Calculator
nav: false
description: An Obsidian plugin that carries measurement uncertainty through a calculation — input forms, significant figures, expanded uncertainty, uncertainty budgets, and Monte Carlo, following the GUM.
---

<div class="row mt-4">
  <div class="col-md-12">
    <p class="lead">
      Uncertainty Calculator is a community plugin for
      <a href="https://obsidian.md/" target="_blank" rel="noopener">Obsidian</a>
      that carries a <code>&plusmn;</code> all the way through a calculation.  Write a
      <code>calc</code> block, enter measured quantities with their uncertainties,
      and reference them in later formulas — the plugin propagates the
      uncertainty (with correct correlations), rounds to significant figures,
      and, on request, reports an expanded uncertainty, an uncertainty budget,
      or a full Monte&nbsp;Carlo distribution.  It grew out of wanting the
      convenience of Python's <code>uncertainties</code>, R's <code>errors</code>,
      or LaTeX's <code>siunitx</code> without leaving my notes.
    </p>
  </div>
</div>

<!-- =========================================================== Links -->
<div class="row text-center mt-4 mb-2">
  <div class="col-md-6 mb-3">
    <a href="obsidian://show-plugin?id=uncertainty-calc"
       class="btn btn-primary btn-lg" style="min-width: 240px;">
      <i class="fas fa-square-root-variable"></i>&nbsp;&nbsp;Open in Obsidian
    </a>
    <p class="mt-2"><small class="text-muted">Opens the plugin page in the app.</small></p>
  </div>
  <div class="col-md-6 mb-3">
    <a href="https://github.com/MasoudMiM/obsidian-uncertainty-calc"
       class="btn btn-primary btn-lg" target="_blank" rel="noopener" style="min-width: 240px;">
      <i class="fab fa-github"></i>&nbsp;&nbsp;Source on GitHub
    </a>
    <p class="mt-2"><small class="text-muted">MIT licensed.</small></p>
  </div>
</div>

<!-- =========================================================== Hero -->
<div class="row mt-4">
  <div class="col-md-12 text-center">
    <img src="{{ '/assets/uncertainty-calc/screenshot.png' | relative_url }}"
         alt="A calc block rendered in Obsidian: assignments with uncertainties, a propagated result, a labelled uncertainty budget, an expanded uncertainty, and an inline result."
         class="img-fluid rounded z-depth-1">
    <p class="mt-2"><small class="text-muted">A <code>calc</code> block rendered in a note: assignments build a running scope, and each line reports its value with a propagated uncertainty.</small></p>
  </div>
</div>

<!-- =========================================================== Usage -->
<div class="row mt-4">
  <div class="col-md-12">
    <h2>How it works</h2>
    <p>
      Add a fenced code block with the language <code>calc</code>.  Each line is a
      comment, an assignment (<code>name = expr</code>), or a bare expression;
      assignments accumulate so later lines can build on earlier results.  A
      trailing <code>| flag</code> asks for more than the bare value —
      <code>budget</code>, <code>expand</code>, or <code>mc</code>.
    </p>
    <pre><code>```calc
# g from a pendulum: Type A period, Type B length
L = 1.000 &plusmn; 0.002
T = typeA(2.01, 2.00, 2.02, 1.99, 2.00)
g = 4 * pi**2 * L / T**2
g | expand
```</code></pre>
    <p>
      Measured quantities can be written as <code>9.81 &plusmn; 0.02</code>, a
      relative <code>200 &plusmn; 2%</code>, the compact <code>1.234(12)</code>,
      a Type&nbsp;A sample <code>typeA(2.01, 2.00, ...)</code> (mean &plusmn;
      standard error), or a Type&nbsp;B bound with a distribution such as
      <code>0.05 [rect]</code>.  In ordinary text, an inline code span that
      begins with <code>=</code>, like
      <code>=2*pi*sqrt((1.0 &plusmn; 0.002)/(9.81 &plusmn; 0.02))</code>,
      is replaced by its value.
    </p>
  </div>
</div>

<!-- =========================================================== Why MC -->
<div class="row mt-4">
  <div class="col-md-12">
    <h2>Why propagate distributions?</h2>
    <p>
      The usual (linear) method approximates a formula by its slope at the
      estimate.  That is exact for sums and scalings, but for a nonlinear
      function it can get both the value and the width wrong.  The
      <code>| mc</code> flag instead samples the full input distributions and
      re-evaluates — and the difference can be large.  Squaring a measurement,
      for example, shifts the mean upward and skews the result, neither of which
      a symmetric <code>&plusmn;</code> can express:
    </p>
  </div>
</div>

<div class="row mt-2">
  <div class="col-md-12 text-center">
    <img src="{{ '/assets/uncertainty-calc/mc-emerge.gif' | relative_url }}"
         alt="Animation: as the number of Monte Carlo trials grows, the true distribution of a squared measurement emerges — shifted above the naive value and right-skewed — while the linear Gaussian stays symmetric."
         class="img-fluid rounded z-depth-1" style="max-width: 640px;">
    <p class="mt-2"><small class="text-muted">The Monte&nbsp;Carlo distribution emerging as trials grow, next to the fixed linear approximation.</small></p>
  </div>
</div>

<div class="row mt-3">
  <div class="col-md-12 text-center">
    <img src="{{ '/assets/uncertainty-calc/mc-vs-linear.png' | relative_url }}"
         alt="Three panels comparing the linear Gaussian to the Monte Carlo distribution: a squared value (right-skewed), a sum of two uniforms (triangular), and sine near its peak (linear underestimates the spread)."
         class="img-fluid rounded z-depth-1">
    <p class="mt-2"><small class="text-muted">Three cases where the linear approximation visibly fails, each against the Monte&nbsp;Carlo distribution from the same inputs.</small></p>
  </div>
</div>

<!-- =========================================================== Features -->
<div class="row mt-4">
  <div class="col-md-12">
    <h2>What's inside</h2>
    <ul>
      <li>
        <strong>A calculation sheet</strong> — assignments build a running
        scope, so a formula can reference earlier measured quantities, like a
        spreadsheet that understands uncertainty.
      </li>
      <li>
        <strong>Every input form</strong> — standard uncertainty
        (<code>&plusmn;</code>), relative (<code>%</code>), compact
        <code>1.234(12)</code>, Type&nbsp;A from repeated readings
        (<code>typeA(&hellip;)</code>), and Type&nbsp;B distribution bounds
        (<code>[rect]</code>, <code>[tri]</code>, <code>[k=2]</code>).
      </li>
      <li>
        <strong>Correlation-correct propagation</strong> — a reused variable
        stays correlated with itself, so <code>x - x</code> is exactly zero,
        while two independent measurements do not spuriously cancel.
      </li>
      <li>
        <strong>Expanded uncertainty</strong> — a coverage factor from the
        Student-<em>t</em> at the Welch&ndash;Satterthwaite effective degrees of
        freedom, so a Type&nbsp;A input with few readings widens the interval as
        it should.
      </li>
      <li>
        <strong>Uncertainty budget</strong> — a table of how much each source
        contributes to the combined uncertainty, with the dominant term flagged.
      </li>
      <li>
        <strong>Monte&nbsp;Carlo (GUM Supplement&nbsp;1)</strong> — propagation
        of distributions for nonlinear or non-Gaussian problems, reporting the
        mean, standard deviation, coverage interval, and skewness.
      </li>
      <li>
        <strong>A tested core</strong> — the numerical engine is an independent
        TypeScript implementation, cross-checked test-for-test against Python's
        <code>uncertainties</code>, <code>pint</code>, <code>scipy</code>, and
        <code>numpy</code>.
      </li>
    </ul>
  </div>
</div>

<!-- =========================================================== Install -->
<div class="row mt-4">
  <div class="col-md-12">
    <h2>Install</h2>
    <p>
      In Obsidian, open <strong>Settings &rarr; Community plugins &rarr;
      Browse</strong>, search for <em>Uncertainty Calculator</em>, install, and
      enable it.  Then add a <code>calc</code> code block to any note.  The
      plugin is free and open source under the MIT license.
    </p>
    <p>
      Found a bug or want another feature?  The
      <a href="https://github.com/MasoudMiM/obsidian-uncertainty-calc/issues" target="_blank" rel="noopener">issue tracker</a>
      is open.
    </p>
  </div>
</div>
