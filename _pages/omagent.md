---
layout: page
permalink: /omagent/
title: omagent
nav: false
nav_order: 6
description: omagent, an open-source headless Python agent that turns a natural-language task into a verified OpenModelica model, closing the loop on physics rather than just compilation.
---

<div class="row mt-4">
  <div class="col-md-12">
    <p class="lead">
      omagent is an open-source, headless Python agent for
      <a href="https://openmodelica.org" target="_blank" rel="noopener">OpenModelica</a>
      that turns a natural-language description into a Modelica model that has been
      <em>compiled, simulated, and verified quantitatively</em>.  Instead of stopping
      at "the code the LLM wrote looks right", omagent checks physics: acceptance
      criteria on the simulation trajectories are checked automatically, and every
      failure, compiler or verifier, is parsed into structured feedback for the next
      attempt.  It is testable by construction: the agent loop runs in unit
      tests with a fake compiler and a scripted LLM, no OpenModelica install needed.
    </p>
  </div>
</div>

<!-- ===========================================================
     Working principle diagram (mermaid)
     =========================================================== -->
<div class="row mt-5">
  <div class="col-md-12">
    <h2>How it works</h2>
    <p>
      The core is a narrow generate and repair loop over the OpenModelica
      compiler.  An LLM produces a complete candidate model, omagent loads it,
      checks it, simulates it, and verifies the trajectories against
      quantitative acceptance criteria.  Anything that fails, compilation,
      balance errors, runtime crashes, or a verifier complaint such as
      "final value of x is 1.93, expected 2.0", is parsed into structured
      diagnostics and returned to the LLM as feedback for the next attempt.
    </p>

<div style="max-width: 720px; margin: 0 auto;">
<svg viewBox="0 0 860 340" xmlns="http://www.w3.org/2000/svg" role="img" class="omdiag" aria-label="omagent agent loop: task prompt turned into an LLM candidate, compiler checks and simulation, result verification, and a structured feedback repair loop.">
  <defs>
    <marker id="omarrow" markerWidth="9" markerHeight="8" refX="8" refY="4" orient="auto" markerUnits="userSpaceOnUse">
      <path d="M1,1 L8,4 L1,7" class="omline"/>
    </marker>
    <filter id="omrough" x="-10%" y="-10%" width="120%" height="120%">
      <feTurbulence type="fractalNoise" baseFrequency="0.02 0.03" numOctaves="2" seed="7" result="n"/>
      <feDisplacementMap in="SourceGraphic" in2="n" scale="3.5"/>
    </filter>
  </defs>

  <g filter="url(#omrough)">
    <!-- ===== main pipeline (top rail) ===== -->
    <g>
      <rect x="20"  y="20" width="170" height="56" rx="12" class="ombox"/>
      <text x="105" y="44" class="omt">Task prompt</text>
      <text x="105" y="66" class="omt oms">(natural language)</text>

      <line x1="190" y1="48" x2="212" y2="48" class="omline" marker-end="url(#omarrow)"/>

      <rect x="212" y="20" width="160" height="56" rx="12" class="ombox"/>
      <text x="292" y="44" class="omt">LLM adapter</text>
      <text x="292" y="66" class="omt oms">propose(model)</text>

      <line x1="372" y1="48" x2="414" y2="48" class="omline" marker-end="url(#omarrow)"/>

      <rect x="414" y="18" width="185" height="60" rx="12" class="ombox"/>
      <text x="506" y="42" class="omt">omc session</text>
      <text x="506" y="66" class="omt oms">checkModel · simulate</text>

      <line x1="599" y1="48" x2="641" y2="48" class="omline" marker-end="url(#omarrow)"/>
      <text x="620" y="34" class="omt oms omhalo">pass</text>

      <rect x="641" y="18" width="185" height="60" rx="28" class="ombox"/>
      <text x="733" y="42" class="omt">verified model</text>
      <text x="733" y="66" class="omt oms">+ transcript JSON</text>
    </g>

    <!-- ===== results and repair rail (second row) ===== -->
    <g>
      <path d="M 506 78 L 506 106 L 105 106 L 105 118" class="omline" marker-end="url(#omarrow)"/>

      <rect x="20"  y="120" width="170" height="56" rx="12" class="ombox"/>
      <text x="105" y="144" class="omt">read results</text>
      <text x="105" y="168" class="omt oms">CSV / Dymola .mat</text>

      <line x1="190" y1="148" x2="232" y2="148" class="omline" marker-end="url(#omarrow)"/>

      <rect x="232" y="120" width="250" height="56" rx="12" class="ombox"/>
      <text x="357" y="144" class="omt">quantitative verifiers</text>
      <text x="357" y="168" class="omt oms">expect_final · bounds · dips</text>

      <line x1="482" y1="148" x2="524" y2="148" class="omline" marker-end="url(#omarrow)"/>

      <rect x="524" y="120" width="200" height="56" rx="12" class="ombox"/>
      <text x="624" y="144" class="omt">structured feedback</text>
      <text x="624" y="168" class="omt oms">summarize_for_llm()</text>

      <line x1="624" y1="176" x2="624" y2="212" class="omline" marker-end="url(#omarrow)"/>

      <rect x="514" y="212" width="220" height="72" rx="12" class="ombox"/>
      <text x="624" y="236" class="omt">omc lookup hints</text>
      <text x="624" y="256" class="omt oms">'Class X not found' -></text>
      <text x="624" y="274" class="omt oms">near-match names</text>

      <path d="M 624 116 L 624 96 L 296 96 L 296 78" class="omline" stroke-dasharray="8 6" marker-end="url(#omarrow)"/>
      <text x="470" y="90" class="omt oms omhalo">repair round 2..N</text>
    </g>
  </g>
</svg>

<style>
  @import url('https://fonts.googleapis.com/css2?family=Caveat:wght@600&display=swap');
  .omdiag { display: block; width: 100%; height: auto; font-family: 'Caveat', cursive; }
  .ombox {
    fill: var(--global-card-bg-color, #fff);
    stroke: var(--global-text-color, #333);
    stroke-width: 2.5;
    stroke-linecap: round;
    stroke-linejoin: round;
  }
  .omline { stroke: var(--global-text-color, #333); fill: none; stroke-width: 2; stroke-linecap: round; }
  .omt    { fill: var(--global-text-color, #333); font-size: 24px; text-anchor: middle; }
  .oms    { font-size: 17px; }
  .omhalo { paint-order: stroke; stroke: var(--global-card-bg-color, #fff); stroke-width: 5px; stroke-linejoin: round; }
</style>
  </div>
</div>

<div class="row mt-3">
  <div class="col-md-12">
    <p>
      The distinctive part is the feedback quality, not the loop itself.
      omc output is parsed into records with severity, source location, and a
      failure kind (syntax, lookup, type, balance, connect, initialization,
      runtime).  The dominant observed failure mode for LLM-generated
      Modelica is stale library knowledge: models that reference MSL 3.2
      names that were renamed in MSL 4.x.  On a "Class X not found" failure,
      omagent asks omc what the parent package actually contains
      (<code>getClassNames</code>) and puts near-miss suggestions into the
      fix prompt, so the model learns "did you mean RotationalEMF?"
      from the compiler itself.
    </p>
  </div>
</div>

<!-- ===========================================================
      Capabilities
      =========================================================== -->
<div class="row mt-5">
  <div class="col-md-12">
    <h2>What's inside</h2>
    <ul>
      <li>
        <strong>Structured omc diagnostics</strong>
        (<code>errors.py</code>).  Parses <code>getErrorString()</code>,
        simulation logs, and OMPython exceptions into records with severity,
        source location, and failure kind.  Both OMPython contracts are
        covered: older versions return and let you read the error string,
        newer ones raise, and both collapse to the same structured failure.
      </li>
      <li>
        <strong>Environment-grounded fix hints</strong>
        (<code>loop.py</code>).  As described above, lookup failures are
        remediated with what the loaded libraries actually contain, surfaced
        through near-match ranking.
      </li>
      <li>
        <strong>Quantitative verification</strong>
        (<code>results.py</code>).  Reads CSV and Dymola-format .mat result
        files.  Verifier factories (<code>expect_final</code>,
        <code>expect_value_at</code>, <code>expect_bounds</code>,
        <code>expect_dips_below</code>) produce human-readable complaints
        that feed straight back into the loop, so tuning, not just repair,
        is verifier-driven.
      </li>
      <li>
        <strong>Benchmark task ladder</strong>
        (<code>tasks.py</code>, <code>runner.py</code>).  Five escalating,
        auto-gradable tasks with full transcript capture, in the format the
        <a href="https://github.com/OpenModelica/OpenModelica/issues/15385"
           target="_blank" rel="noopener">OpenModelica benchmark discussion</a>
        calls for.  <code>run_comparison()</code> benchmarks several backends
        with repeated runs and aggregates pass rates, attempt statistics, and
        variance.
      </li>
      <li>
        <strong>Warning-level quality gates</strong>.  Some omc diagnostics
        come as warnings yet mean the model is sloppy.  Gated warnings
        (initial conditions not fully specified, inconsistent units,
        over/under-determined systems) become verifier-style complaints that
        reach the LLM, while benchmark scores stay comparable because the
        gate is opt-in.
      </li>
      <li>
        <strong>LLM-backend-agnostic</strong>
        (<code>llm.py</code>).  The loop depends on a one-method protocol,
        <code>propose(task, previous_code, error_summary)</code>.  Adapters
        ship for Anthropic and for any OpenAI-compatible endpoint, meaning
        Ollama, LM Studio, llama.cpp and vLLM, including local open-weight
        models, work out of the box.
      </li>
    </ul>
  </div>
</div>

<!-- ===========================================================
      Install
      =========================================================== -->
<div class="row mt-5">
  <div class="col-md-12">
    <h2>Install</h2>
    <p>
      Requires Python 3.10 or newer.  The core package has no hard
      dependencies; features are opt-in extras:
    </p>
<pre><code>pip install omagent            # parsers + loop only (no omc needed)
pip install "omagent[omc]"     # + OMPython (talk to a real omc)
pip install "omagent[results]" # + scipy (.mat result files; CSV needs nothing)
pip install "omagent[all]"     # everything</code></pre>
    <p>
      To simulate you also need
      <a href="https://openmodelica.org" target="_blank" rel="noopener">OpenModelica</a>
      (tested with 1.26 and 1.27) with the Modelica Standard Library
      installed for the headless compiler.
    </p>
  </div>
</div>

<!-- ===========================================================
      Quick start
      =========================================================== -->
<div class="row mt-5">
  <div class="col-md-12">
    <h2>Quick start</h2>
    <p>
      In a few lines, an agent loop with physics acceptance criteria and a
      running feedback loop:
    </p>
<pre><code>from omagent import AgentLoop, OMSession, all_of
from omagent.results import expect_bounds, expect_final
from omagent.llm import OpenAICompatLLM

verifier = all_of(
    expect_bounds("x", lo=-0.105, hi=0.105),
    expect_final("x", 0.0, atol=0.06),
)

loop = AgentLoop(
    OMSession(),
    OpenAICompatLLM(model="qwen2.5-coder:14b"),  # or ClaudeLLM()
    max_attempts=4,
    simulate_options={"stopTime": 10.0, "outputFormat": "csv"},
    verifier=verifier,
)

result = loop.run(
    "A mass-spring-damper: m = 1 kg, c = 100 N/m, d = 1 N.s/m, released "
    "from x = 0.1 m at rest. Name position x and velocity v.")

print(result.success, result.model_name)
for attempt in result.attempts:
    print(attempt.n, attempt.stage, attempt.complaint)</code></pre>
    <p>
      Any object with a <code>propose</code> method works as the LLM, so the
      backend is yours to choose: hosted frontier models, local coders, or
      a hand-rolled proxy.
    </p>
  </div>
</div>

<!-- ===========================================================
      Design notes
      =========================================================== -->
<div class="row mt-5">
  <div class="col-md-12">
    <h2>Design notes</h2>
    <ul>
      <li>
        <strong>Testable by construction.</strong>
        <code>OMSession</code> talks to any object with
        <code>sendExpression()</code>; tests replay recorded omc output, so
        the full agent loop runs in unit tests without a compiler or an API
        key.  The 101 unit tests finish in well under a second.
      </li>
      <li>
        <strong>Both OMPython contracts are handled.</strong>  Older
        OMPython returns and lets the caller read
        <code>getErrorString()</code>; newer versions raise
        <code>OMCSessionException</code> on error-level messages.  Both paths
        produce identical structured failures, never uncaught exceptions.
      </li>
      <li>
        <strong>Environment failures are not model failures.</strong>
        The ladder runner loads the MSL when a task requires it and reports
        load problems as <code>environment</code> outcomes with zero attempts
        charged to the LLM.
      </li>
    </ul>
  </div>
</div>

<!-- ===========================================================
      Links
      =========================================================== -->
<div class="row mt-5">
  <div class="col-md-12">
    <h2>Where to get it</h2>
    <p>
      Code and documentation:
      <a href="https://github.com/MasoudMiM/omagent" target="_blank" rel="noopener">github.com/MasoudMiM/omagent</a>.
      The benchmark transcripts, successes and failures alike, are the
      seed material for the OpenModelica benchmark discussion, and the
      project is BSD-3-Clause licensed.
    </p>
  </div>
</div>

