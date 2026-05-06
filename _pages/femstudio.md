---
layout: page
permalink: /femstudio/
title: FEM Studio
nav: true
nav_order: 5
description: A modern desktop GUI for the Elmer FEM solver, born from research frustration and shared in case it's useful to anyone else.
---

{% comment %}
  Page tone: personal & humble. Two-tier naming: brand "FEM Studio"
  on the website, binary "ElmerStudio" in downloads / window title /
  --version output. Screenshots lead each section so users can see
  the actual app before reading prose about it.

  Note for future-me: Jekyll evaluates Liquid tags inside HTML
  comments (it parses Liquid before HTML). Use {% raw %}{ % comment %}{% endraw %}
  blocks (like this one) for documentation that mentions other
  Liquid tags — that way the parser just skips the whole region.
{% endcomment %}

<div class="row mt-4">
  <div class="col-md-12">
    <p class="lead">
      FEM Studio is a desktop interface for the
      <a href="http://www.elmerfem.org/" target="_blank" rel="noopener">Elmer FEM</a>
      solver.  It started as a personal project — I kept hitting the same
      friction every time I picked Elmer back up for research, and at some
      point I decided to fix it for myself.  This is what came out.  Sharing
      it here in case it saves you the same trips back to the docs.
    </p>
  </div>
</div>

<!-- ===========================================================
     Downloads
     =========================================================== -->
<div class="row mt-4">
  <div class="col-md-12">
    <h2>Download</h2>
    <p>
      Free for personal, academic, and commercial use.  Closed-source
      binary, no account required.  See the
      <a href="https://github.com/FEMStudio/femstudio-releases/blob/main/LICENSE.md" target="_blank" rel="noopener">license</a>
      for the small print.
    </p>

    <div class="row text-center mt-4 mb-4">
      <div class="col-md-6 mb-3">
        <a href="https://github.com/FEMStudio/femstudio-releases/releases/latest"
           class="btn btn-primary btn-lg" target="_blank" rel="noopener"
           style="min-width: 240px;">
          <i class="fab fa-linux"></i>&nbsp;&nbsp;Linux (AppImage)
        </a>
        <p class="mt-2"><small class="text-muted">Ubuntu 24.04+ tested.  ~228&nbsp;MB.</small></p>
      </div>
      <div class="col-md-6 mb-3">
        <a href="https://github.com/FEMStudio/femstudio-releases/releases/latest"
           class="btn btn-primary btn-lg" target="_blank" rel="noopener"
           style="min-width: 240px;">
          <i class="fab fa-windows"></i>&nbsp;&nbsp;Windows (Installer)
        </a>
        <p class="mt-2"><small class="text-muted">Windows 10 / 11.  ~250&nbsp;MB.</small></p>
      </div>
    </div>

    <p class="text-center">
      <a href="https://github.com/FEMStudio/femstudio-releases/releases/latest" target="_blank" rel="noopener">
        See all release artifacts on GitHub
      </a>
      &nbsp;·&nbsp;
      <img src="https://img.shields.io/github/downloads/FEMStudio/femstudio-releases/total.svg?style=flat&label=Total downloads&color=blue"
           alt="Total downloads" style="vertical-align: middle;">
    </p>

    <p class="text-center">
      <small class="text-muted">
        The downloaded files are named <code>ElmerStudio-*</code> —
        that's the binary's internal name, which also appears in the
        application window title.  Same software, no surprises.
      </small>
    </p>
  </div>
</div>

<!-- ===========================================================
     Why this exists
     =========================================================== -->
<div class="row mt-5">
  <div class="col-md-12">
    <h2>Why this exists</h2>
    <p>
      I used Elmer for research and kept running into the same friction.
      Elmer's solver is genuinely excellent — decades of careful
      development by CSC and the Finnish FEM community, free, open
      source, and capable of holding its own against commercial codes.
      But every time I onboarded a student, the first hour with Elmer
      was spent on workflow rather than physics: setting up a problem
      meant editing the Solver Input File (.sif) by hand, mesh
      generation meant flipping between separate command-line tools,
      tweaking a parameter to study a trend meant duplicating the whole
      project, and viewing results meant launching ParaView, which
      itself has been
      <a href="https://discourse.paraview.org/t/im-using-ubuntu-and-i-am-unable-to-open-a-vtp-vtu-vti-file-that-was-working-previously/15517" target="_blank" rel="noopener">broken on Ubuntu 24.04+</a>
      since early 2024 because of an upstream libexpat regression.
    </p>
    <p>
      I built FEM Studio first to remove those rough edges from my own
      workflow, then realised it might help students and newcomers get
      to the interesting parts of FEM faster.  ElmerGUI, the official
      graphical frontend, remains a great option and a complementary
      tool — FEM Studio simply takes a different approach in places
      where I wanted something different.  Below is what's actually
      inside.
    </p>
  </div>
</div>

<!-- ===========================================================
     Quick start: welcome screen
     =========================================================== -->
<div class="row mt-5">
  <div class="col-md-12">
    <h2>Start fast: pick a template, hit go</h2>
    <p>
      Open the app and you're already a click away from a working
      simulation.  Five quick-start templates cover the most common
      Elmer workflows — Heat Conduction, Fluid Flow, Structural
      Mechanics, Electrostatics — plus a Custom Setup wizard for
      everything else.  Each template scaffolds a complete SIF with
      sensible defaults that you can run immediately, then refine.
    </p>
  </div>
</div>

<div class="row mt-3">
  <div class="col-md-12">
    <img src="{{ '/assets/img/femstudio/welcome.png' | relative_url }}"
         alt="FEM Studio welcome screen showing five simulation templates and the Simulation Setup form pre-populated with defaults"
         class="img-fluid rounded z-depth-1">
    <p class="text-center mt-2">
      <small class="text-muted">
        The welcome screen.  Pick a template card or open an existing
        <code>.sif</code> file.  Common simulation parameters sit in
        the dock on the left, ready to edit.
      </small>
    </p>
  </div>
</div>

<!-- ===========================================================
     Mesh generator
     =========================================================== -->
<div class="row mt-5">
  <div class="col-md-12">
    <h2>Built-in parametric mesh generator</h2>
    <p>
      Mesh creation is integrated.  Nine geometry templates (1D line,
      2D rectangles, L-shapes, layered slabs, axisymmetric cylinders,
      polar shells, 3D boxes, hollow and solid cylinders) cover most
      academic and prototyping cases.  Pick one, dial in the
      parameters, and a 3D preview updates as you type — no exporting
      to a separate mesh file format and re-importing.
    </p>
    <p>
      The 2D paint canvas lets you assign material regions and
      boundary IDs visually before the mesh is ever generated.
      Coordinate-system conversions (revolve a 2D section into an
      axisymmetric solid, extrude into 3D) are done through five
      mapping modes that ship with the app, with no command-line
      ElmerGrid invocations to remember.
    </p>
  </div>
</div>

<div class="row mt-3">
  <div class="col-md-12">
    <img src="{{ '/assets/img/femstudio/mesh_generator.png' | relative_url }}"
         alt="Mesh generator with subcell controls on the left, 2D paint canvas on the right, and a live 3D preview of the resulting cylinder mesh below"
         class="img-fluid rounded z-depth-1">
    <p class="text-center mt-2">
      <small class="text-muted">
        Mesh generator: parametric controls, 2D paint canvas for
        material regions, and a live 3D preview that updates as you
        change parameters.
      </small>
    </p>
  </div>
</div>

<div class="row mt-3">
  <div class="col-md-12">
    <p>
      For the cases the templates don't cover, the
      <strong>Mesh → Operations</strong> menu exposes ElmerGrid's
      full toolbox through a GUI: Scale, Translate, Rotate, Merge
      Nodes, Reduce Order, Clone, Mirror.  Each operation is a
      simple form-based dialog that runs ElmerGrid for you with the
      right flags — no need to remember the
      <code>-scale</code> / <code>-rotate</code> / <code>-merge</code>
      command-line syntax.  External meshes (Gmsh
      <code>.msh</code>, Abaqus <code>.unv</code>, STL) load
      directly.
    </p>
  </div>
</div>

<!-- ===========================================================
     Editor + outline + integrated results
     =========================================================== -->
<div class="row mt-5">
  <div class="col-md-12">
    <h2>One window, full workflow</h2>
    <p>
      The main editor is built around a single principle: everything
      you need to set up, run, and inspect a simulation lives in one
      window, not five.  The semantic outline on the left groups the
      SIF into Setup, Physics, Domain, and Boundaries — click any
      entry to jump to it.  The middle pane is a syntax-highlighted
      SIF editor that round-trips losslessly with the form-based
      dock panels.  The right pane is an integrated VTK results
      viewer with field selection, colormaps (jet, viridis, plasma,
      and more), displacement warping, and clip planes.
    </p>
    <p>
      When you do need ParaView's full power, one click opens the
      current results in it.  The "Open in ParaView" path also
      transparently re-encodes the VTU on the way out to dodge the
      libexpat 2.6.0–2.6.2 bug that breaks ParaView on Ubuntu
      24.04+ — so it just works on a stock install with no patching.
    </p>
  </div>
</div>

<div class="row mt-3">
  <div class="col-md-12">
    <img src="{{ '/assets/img/femstudio/full_editor.png' | relative_url }}"
         alt="Full editor view: outline tree on the left, syntax-highlighted SIF in the middle, integrated VTK results viewer on the right showing displacement field on a meshed solid"
         class="img-fluid rounded z-depth-1">
    <p class="text-center mt-2">
      <small class="text-muted">
        Outline tree, SIF editor, and integrated results viewer
        all in one window.  This particular project couples linear
        elasticity with a heat equation; the right pane is showing
        the displacement field on a refined mesh of about 67k nodes
        and 15k cells.
      </small>
    </p>
  </div>
</div>

<!-- ===========================================================
     Parameter sweeps
     =========================================================== -->
<div class="row mt-5">
  <div class="col-md-12">
    <h2>Parameter sweeps as a first-class feature</h2>
    <p>
      Studying how a result depends on material properties, boundary
      conditions, or geometry is something Elmer users do constantly
      — and historically meant either duplicating the project N
      times or writing shell scripts that templated the SIF.  FEM
      Studio makes parameter sweeps a built-in feature.  Drop a
      sibling <code>&lt;case&gt;.sweep.yaml</code> file next to your
      SIF, declare which keywords vary and over what values, and
      hit <strong>Run sweep</strong>.
    </p>
    <p>
      Three sweep modes ship out of the box:
    </p>
    <ul>
      <li>
        <strong>Cartesian product</strong> — every combination of
        every parameter.  Use it for full-factorial studies.
      </li>
      <li>
        <strong>Zip</strong> — parameters advance together in
        lockstep.  Use it when you want N pre-paired
        configurations rather than N×M combinations.
      </li>
      <li>
        <strong>Latin hypercube</strong> — random samples spread
        across the parameter space.  Use it for design-of-experiments
        style coverage when full Cartesian would be too many runs.
      </li>
    </ul>
    <p>
      Each sweep step runs in its own subdirectory with its own
      generated SIF, runs the solver, and surfaces results in a
      live status table as the sweep progresses.  You can abort
      the whole sweep, skip just the current step, or watch
      convergence plot live for the active run.  When it's done,
      you have a structured directory tree of every result, ready
      to post-process or feed into your own analysis script.
    </p>
  </div>
</div>

<!-- ===========================================================
     Other features worth mentioning
     =========================================================== -->
<div class="row mt-5">
  <div class="col-md-12">
    <h2>Other things you get</h2>
    <ul>
      <li>
        <strong>Live convergence plotting</strong> while ElmerSolver
        runs.  Residuals stream into a chart in real time, so you
        see divergence the moment it happens instead of staring at
        terminal output.
      </li>
      <li>
        <strong>A 35-entry materials library</strong> with one-click
        insert.  Steel, aluminum, copper, water, air, and the
        usual suspects all pre-defined with units handled for you.
      </li>
      <li>
        <strong>Pre-flight validation</strong> before each run.
        Catches things like two Body sections claiming the same
        mesh ID (Elmer silently honors only the first) and unmapped
        boundary conditions before you find them at simulation
        time.
      </li>
      <li>
        <strong>Parallel solver support</strong> through a settings
        dialog — pick the number of MPI processes and FEM Studio
        invokes <code>ElmerSolver_mpi</code> with the right
        partitioning.
      </li>
      <li>
        <strong>Cross-platform</strong>: same UI, same project
        files, same workflow on both Linux and Windows.  Modern
        Qt 6 stack, no Qt 4 build battles on current distros.
      </li>
      <li>
        <strong>Self-contained binary</strong>.  Linux AppImage and
        Windows installer ship with everything they need.  No pip,
        no virtualenv, no PySide6 dependency hell.
      </li>
    </ul>
  </div>
</div>

<!-- ===========================================================
     Honest boundaries
     =========================================================== -->
<div class="row mt-5">
  <div class="col-md-12">
    <h2>What it isn't</h2>
    <ul>
      <li>
        <strong>It isn't the solver.</strong>  FEM Studio is a frontend.
        You need ElmerSolver itself installed and on your PATH for
        simulations to run.  Get it from
        <a href="http://www.elmerfem.org/blog/binaries/" target="_blank" rel="noopener">elmerfem.org</a>
        or your distribution's package manager.
      </li>
      <li>
        <strong>It isn't a CAD tool.</strong>  The mesh generator
        covers parametric primitives well but doesn't do general
        CAD import.  For complex geometries, mesh externally with
        Gmsh, Salome, or your CAD tool's mesh export, then load
        the <code>.msh</code> / <code>.unv</code> / <code>.stl</code>
        file into FEM Studio.
      </li>
      <li>
        <strong>It isn't every Elmer feature.</strong>  Elmer's
        solver has decades of accumulated capability.  FEM Studio
        ships eight solver definition files covering common
        physics (heat, linear elasticity, Navier-Stokes,
        electrostatics, Helmholtz, advection-diffusion, static
        current, result output); the rest is still accessible by
        editing the SIF directly in the integrated text view.  If
        a feature you need would benefit from a structured form,
        <a href="https://github.com/FEMStudio/femstudio-releases/issues" target="_blank" rel="noopener">file an issue</a>.
      </li>
      <li>
        <strong>It isn't open source.</strong>  Free as in beer, not
        as in speech.  See the license for what's allowed and what
        isn't.  This may change in the future.
      </li>
    </ul>
  </div>
</div>

<!-- ===========================================================
     Feedback
     =========================================================== -->
<div class="row mt-5">
  <div class="col-md-12">
    <h2>Feedback &amp; bug reports</h2>
    <p>
      The public release repository on GitHub doubles as the issue
      tracker:
      <a href="https://github.com/FEMStudio/femstudio-releases/issues" target="_blank" rel="noopener">github.com/FEMStudio/femstudio-releases/issues</a>.
      Bug reports, feature ideas, "this menu is confusing"
      observations, "what does this option mean" questions — all
      welcome.  When reporting a bug, please include your OS, the
      FEM Studio version (Help → About in the app, or run with
      <code>--version</code>), and what you were doing when it
      happened.
    </p>
    <p>
      I read everything but reply on a rolling basis — this is a side
      effort, not a full-time gig.
    </p>
  </div>
</div>
