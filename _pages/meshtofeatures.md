---
layout: page
permalink: /meshtofeatures/
title: MeshToFeatures
nav: false
description: A FreeCAD workbench that reverse-engineers STL meshes of prismatic parts back into editable PartDesign bodies — surface recognition, design-intent snapping, and build-history reconstruction.
---

<div class="row mt-4">
  <div class="col-md-12">
    <p class="lead">
      MeshToFeatures is a workbench for
      <a href="https://www.freecad.org/" target="_blank" rel="noopener">FreeCAD</a>
      that takes a triangle mesh — the STL you downloaded, were handed, or
      whose original CAD file is long gone — and rebuilds it as a native,
      fully editable PartDesign body.  It started with a familiar
      frustration: an STL is a dead end.  You can print it, but you cannot
      resize a hole, deepen a pocket, or move a boss.  This workbench walks
      the road back: it recognizes the analytic surfaces hiding in the
      triangles, snaps their parameters to the design intent they came
      from, infers a plausible build history, and hands you real sketches,
      pads, pockets, and holes you can edit.
    </p>
  </div>
</div>

<!-- =========================================================== Links -->
<div class="row text-center mt-4 mb-2">
  <div class="col-md-6 mb-3">
    <a href="https://github.com/MasoudMiM/MeshToFeatures"
       class="btn btn-primary btn-lg" target="_blank" rel="noopener" style="min-width: 240px;">
      <i class="fab fa-github"></i>&nbsp;&nbsp;Source on GitHub
    </a>
    <p class="mt-2"><small class="text-muted">LGPL-2.1-or-later.</small></p>
  </div>
  <div class="col-md-6 mb-3">
    <a href="https://github.com/MasoudMiM/MeshToFeatures/releases"
       class="btn btn-primary btn-lg" target="_blank" rel="noopener" style="min-width: 240px;">
      <i class="fas fa-download"></i>&nbsp;&nbsp;Download
    </a>
    <p class="mt-2"><small class="text-muted">v0.17.1 — bug reports with the STL attached are welcome.</small></p>
  </div>
</div>

<!-- =========================================================== Gallery -->
<div class="row mt-4">
  <div class="col-md-12">
    <h2>From triangles to a feature tree</h2>
    <p>
      Each rebuild produces two things: a reconstruction group holding the
      recognized analytic surfaces, and a PartDesign body whose feature
      tree reads like the part's manufacturing story — terraces, hole
      patterns, counterbores, countersinks, conical pockets, cross-holes,
      lateral pads.
    </p>
  </div>
</div>

<div class="row mt-2">
  <div class="col-md-4 col-12 mb-4 text-center">
    <img src="{{ '/assets/meshtofeatures/featuretype.png' | relative_url }}"
         alt="A machined test part rebuilt as a full PartDesign feature tree"
         class="img-fluid rounded z-depth-1">
    <p class="mt-2"><small class="text-muted">A machined test part rebuilt
    feature by feature: fifteen terraces, an 8&times; counterbored hole grid,
    a cross-hole, and a beveled lateral pad — all editable.</small></p>
  </div>
  <div class="col-md-4 col-12 mb-4 text-center">
    <img src="{{ '/assets/meshtofeatures/tray-holes.png' | relative_url }}"
         alt="Hole patterns recognized and labeled with diameters and bolt-circle data"
         class="img-fluid rounded z-depth-1">
    <p class="mt-2"><small class="text-muted">Pattern recognition with
    design-intent labels: bolt circles with their BCDs, and 3.3&nbsp;mm
    holes identified as M4 tap drills.</small></p>
  </div>
  <div class="col-md-4 col-12 mb-4 text-center">
    <img src="{{ '/assets/meshtofeatures/rotated-cube.png' | relative_url }}"
         alt="A part in arbitrary orientation reconstructed correctly"
         class="img-fluid rounded z-depth-1">
    <p class="mt-2"><small class="text-muted">Orientation is detected, not
    assumed: a rotated part reconstructs in its own working frame.</small></p>
  </div>
</div>

<!-- =========================================================== How -->
<div class="row mt-4">
  <div class="col-md-12">
    <h2>How it works</h2>
    <p>
      The pipeline has four stages.  <strong>Segmentation &amp;
      recognition</strong> region-grows the mesh into patches and fits
      analytic surfaces (planes, cylinders, cones) to each.
      <strong>Snapping</strong> then nudges the raw fits toward design
      intent — near-parallel directions unify to canonical axes, coaxial
      cylinders merge, near-equal radii equalize, and values round to
      grid-friendly numbers — with every decision logged to an audit trail,
      and a guard that reverts any snap the mesh contradicts.
      <strong>Feature detection</strong> turns surfaces into semantics:
      holes, counterbores, pockets, steps, bosses, patterns.  Finally the
      <strong>executor</strong> replays the inferred history as native
      PartDesign operations, with per-feature failure recovery so one
      stubborn boolean degrades a single feature — loudly — rather than the
      whole part.
    </p>
    <p>
      The geometry core is FreeCAD-free (numpy / scipy / trimesh /
      shapely), which is what makes the 450+ test pytest suite and the
      multi-part regression gate possible outside the GUI.
    </p>
  </div>
</div>

<!-- =========================================================== Scope -->
<div class="row mt-4">
  <div class="col-md-12">
    <h2>What it handles — and what it doesn't</h2>
    <p>
      The sweet spot is <strong>prismatic parts</strong>: plates, brackets,
      housings, fixtures — the world of planes, cylinders, and cones.  Within
      that scope it rebuilds base solids from arbitrary footprints,
      multi-depth stepped pockets with islands, through and blind holes,
      grid patterns, counterbores (including bores opening below the top
      face), countersunk and counterdrilled holes, conical pockets and
      tapered through holes, cross-axis holes both through and blind,
      chamfers, and lateral flanges and gussets with true-slope
      undersides.
    </p>
    <p>
      Honest limits: organic and 3D-scanned shapes are out of scope; spheres
      and tori are fitted by the core but not yet rebuilt as features; and
      dimensional fidelity is bounded by the mesh tessellation (snapping
      works at roughly 0.1% of the part diagonal).  The
      <a href="https://github.com/MasoudMiM/MeshToFeatures#limitations-please-read-before-filing-bugs"
         target="_blank" rel="noopener">README</a> spells all of this out.
    </p>
  </div>
</div>

<!-- =========================================================== Install -->
<div class="row mt-4">
  <div class="col-md-12">
    <h2>Install</h2>
    <p>
      In FreeCAD (1.1+), install <em>MeshToFeatures</em> from the
      <strong>Addon Manager</strong> and restart (until it appears in the
      official addon index, add
      <code>https://github.com/MasoudMiM/MeshToFeatures</code> as a custom
      repository under <strong>Edit → Preferences → Addon
      Manager</strong> first).  The geometry core needs a few Python packages
      inside FreeCAD's interpreter (numpy, scipy, trimesh, shapely) — the
      Addon Manager will offer them, and the
      <a href="https://github.com/MasoudMiM/MeshToFeatures/blob/main/freecad/meshtofeatures_wb/docs/VERIFY.md"
         target="_blank" rel="noopener">verification guide</a> covers each
      platform.
    </p>
    <p>
      If a part reconstructs badly, the most useful
      bug report in the world is the STL itself plus the Report view
      output — the
      <a href="https://github.com/MasoudMiM/MeshToFeatures/issues"
         target="_blank" rel="noopener">issue tracker</a> is open.
    </p>
  </div>
</div>