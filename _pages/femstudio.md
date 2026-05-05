---
layout: page
permalink: /femstudio/
title: FEM Studio
nav: true
nav_order: 5
description: A modern desktop GUI for Elmer FEM, born from research frustration and shared in case it's useful to anyone else.
---

<!--
  Page tone: personal & humble.  Two-tier naming convention adopted
  for FEM Studio: "FEM Studio" everywhere user-facing on the web,
  while the underlying binary keeps its original "ElmerStudio" name
  (which is what they'll see in their downloads folder, --version
  output, and the application window title bar).  The third paragraph
  of the "Why this exists" section names this explicitly so observant
  users aren't confused.
-->

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
     ===========================================================
     Direct links to the public release repo.  GitHub's
     /releases/latest endpoint redirects to the most recent
     release, so we don't have to update version numbers here
     when shipping a new build.

     Note: the BINARY filenames still have "ElmerStudio-" prefix
     (that's the internal name of the underlying executable).
     Users see this in their downloads folder.  Reassuring them
     in the file-name notes column avoids confusion.
-->
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

<!-- Optional: a hero screenshot.  Drop your screenshot at
     assets/img/femstudio/hero.png and uncomment.  The figure.liquid
     include is al-folio's standard image helper.

<div class="row mt-4">
  <div class="col-md-12">
    {% include figure.liquid path="assets/img/femstudio/hero.png" class="img-fluid rounded z-depth-1" zoomable=true caption="The main editor view, showing the SIF outline, structured form panel, and integrated mesh / results viewers." %}
  </div>
</div>
-->

<!-- ===========================================================
     Why this exists
     =========================================================== -->
<div class="row mt-5">
  <div class="col-md-12">
    <h2>Why this exists</h2>
    <p>
      I used Elmer for research and kept running into the same wall:
      Elmer's solver is genuinely excellent — decades of careful
      development by CSC and the Finnish FEM community, free, open
      source, and with capabilities that hold their own against
      commercial codes — but the GUI hadn't aged well.  ElmerGUI is
      built on Qt 4, which has been end-of-life since 2015; on
      modern Linux distributions it often won't compile at all
      without manual intervention.  More fundamentally, as
      Elmer's own maintainers have noted on the project forum,
      ElmerGUI represents about
      "<a href="http://www.elmerfem.org/forum/viewtopic.php?f=4&t=3485" target="_blank" rel="noopener">one man-year of work</a>"
      and never fully kept up with the solver.
    </p>
    <p>
      That meant every time I onboarded a student, the first hour of
      working with Elmer was spent on the workflow rather than the
      physics.  Setting up a problem meant editing the Solver Input
      File (.sif) by hand, mostly without help from the GUI for
      anything beyond the most common solvers.  Mesh generation
      meant launching a separate tool, exporting in the right
      format, importing again, and hoping the boundary numbers
      lined up.  Looking at results meant launching ParaView,
      which itself has been broken on Ubuntu 24.04+
      <a href="https://discourse.paraview.org/t/im-using-ubuntu-and-i-am-unable-to-open-a-vtp-vtu-vti-file-that-was-working-previously/15517" target="_blank" rel="noopener">because of an upstream libexpat regression</a>.
    </p>
    <p>
      I built FEM Studio first to remove those rough edges from my own
      workflow.  Then I realised it might help students and newcomers
      get to the interesting parts of FEM faster, without having to
      become a SIF-format expert before solving their first problem.
      So here it is.
    </p>
  </div>
</div>

<!-- ===========================================================
     What's different
     ===========================================================
     Honest comparison with ElmerGUI.  No marketing-speak.
     Each row is something I can stand behind based on real
     differences between the two tools.
-->
<div class="row mt-5">
  <div class="col-md-12">
    <h2>What's different from ElmerGUI</h2>
    <p>
      FEM Studio is not a fork or a replacement of ElmerGUI — both are
      frontends for the same Elmer solver underneath.  ElmerGUI works
      well for many people and remains the official option.  FEM Studio
      exists because there are specific places where I wanted something
      different:
    </p>

    <table class="table">
      <thead>
        <tr>
          <th style="width: 30%;">Area</th>
          <th>ElmerGUI</th>
          <th>FEM Studio</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td><strong>Toolkit</strong></td>
          <td>Qt 4 (end-of-life since 2015; build issues on modern distros)</td>
          <td>Qt 6 / PySide6, ships as a self-contained binary</td>
        </tr>
        <tr>
          <td><strong>Mesh creation</strong></td>
          <td>Imports an external mesh; integrated meshing relies on a separate tool launch</td>
          <td>Built-in template-based mesh generator with live preview (rectangle, layered slab, box, cylinder, etc.)</td>
        </tr>
        <tr>
          <td><strong>SIF editing</strong></td>
          <td>Form-based for supported solvers; manual text editing for everything else</td>
          <td>Form-based panels backed by a structured outline of every section, with syntax-highlighted text view that round-trips losslessly</td>
        </tr>
        <tr>
          <td><strong>Materials</strong></td>
          <td>Add by hand or via a small built-in library</td>
          <td>Searchable materials library with units handling and direct copy-into-current-project</td>
        </tr>
        <tr>
          <td><strong>Results viewing</strong></td>
          <td>External (ParaView)</td>
          <td>Integrated VTK-based viewer for quick inspection; "Open in ParaView" button when you need full power</td>
        </tr>
        <tr>
          <td><strong>Parameter sweeps</strong></td>
          <td>Manual — duplicate the project for each variant</td>
          <td>First-class feature: define a sweep, run all variants, compare convergence side-by-side</td>
        </tr>
        <tr>
          <td><strong>"Open in ParaView"</strong></td>
          <td>Direct launch (broken on systems with libexpat 2.6.0–2.6.2)</td>
          <td>Re-encodes the VTU on the fly to dodge the libexpat bug, so it just works on Ubuntu 24.04+ out of the box</td>
        </tr>
      </tbody>
    </table>

    <p>
      <small>
        If you're a long-time ElmerGUI user with custom solver definitions,
        complex geometry imports, or a heavy parallel-MPI workflow, I'd
        suggest sticking with what works.  FEM Studio's sweet spot is
        students, newcomers, and anyone who wants a faster path from
        problem statement to first solution.
      </small>
    </p>
  </div>
</div>

<!-- ===========================================================
     Note on what FEM Studio is NOT
     ===========================================================
     This section sets honest expectations.  Better to say what's
     missing up front than to have users discover gaps mid-workflow.
-->
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
        <strong>It isn't a CAD tool.</strong>  FEM Studio's mesh generator
        covers parametric primitives well but doesn't do general CAD
        import.  For complex geometries, mesh externally with Gmsh,
        Salome, or your CAD tool's mesh export, then load the .msh /
        .unv / .stl file into FEM Studio.
      </li>
      <li>
        <strong>It isn't every Elmer feature.</strong>  Elmer's solver
        has decades of accumulated capability.  FEM Studio surfaces the
        common 80% with structured forms; the remaining 20% is still
        accessible by editing the SIF directly in the integrated text
        view.  If a feature you need is buried in the text view and
        you'd like a form for it,
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
     ===========================================================
     Issue tracker on the public release repo doubles as the
     bug/feature/feedback channel.  Issue creation requires a
     GitHub account — most users in this audience already have one,
     but if not, an alternative low-friction channel via the site's
     contact info would be a future addition.
-->
<div class="row mt-5">
  <div class="col-md-12">
    <h2>Feedback &amp; bug reports</h2>
    <p>
      The public release repository on GitHub doubles as the issue
      tracker:
      <a href="https://github.com/FEMStudio/femstudio-releases/issues" target="_blank" rel="noopener">github.com/FEMStudio/femstudio-releases/issues</a>.
      Bug reports, feature ideas, "this menu is confusing"
      observations, "what does this option mean" questions — all
      welcome.  When reporting a bug, please include your OS,
      the FEM Studio version (Help → About in the app, or run with
      <code>--version</code>), and what you were doing when it
      happened.
    </p>
    <p>
      I read everything but reply on a rolling basis — this is a side
      effort, not a full-time gig.
    </p>
  </div>
</div>
