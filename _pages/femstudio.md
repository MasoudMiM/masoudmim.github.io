---
layout: page
permalink: /femstudio/
title: FEM Studio
nav: true
nav_order: 5
description: ElmerStudio — a modern desktop GUI for the Elmer FEM solver, born from research frustration and shared in case it's useful to anyone else.
---

<div class="row mt-4">
  <div class="col-md-12">
    <p class="lead">
      ElmerStudio is a desktop interface for the
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
        <p class="mt-2"><small class="text-muted">Ubuntu 24.04+ tested.</small></p>
      </div>
      <div class="col-md-6 mb-3">
        <a href="https://github.com/FEMStudio/femstudio-releases/releases/latest"
           class="btn btn-primary btn-lg" target="_blank" rel="noopener"
           style="min-width: 240px;">
          <i class="fab fa-windows"></i>&nbsp;&nbsp;Windows (Installer)
        </a>
        <p class="mt-2"><small class="text-muted">Windows 10 / 11.</small></p>
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
      I built ElmerStudio first to remove those rough edges from my own
      workflow, then realised it might help students and newcomers get
      to the interesting parts of FEM faster.  ElmerGUI, the official
      graphical frontend, remains a great option and a complementary
      tool — ElmerStudio simply takes a different approach in places
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
      simulation.  Four quick-start cards cover the most common
      Elmer workflows — Heat Conduction, Fluid Flow, Structural
      Mechanics, Electrostatics — each pre-wired to a sensible
      solver and starting material.  A fifth card opens a four-step
      Custom Setup wizard that walks you through picking physics,
      simulation type and time-stepping, an existing mesh
      directory, and an initial material from the library.  Either
      way, you end up with a complete SIF that you can run
      immediately and refine from there.
    </p>
  </div>
</div>

<div class="row mt-3">
  <div class="col-md-12">
    <img src="{{ '/assets/img/femstudio/welcome.png' | relative_url }}"
         alt="ElmerStudio welcome screen showing four physics templates plus a Custom Setup wizard, with the Simulation Setup form pre-populated with defaults"
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
      Mesh creation is integrated.  Eleven geometry templates
      (Segment 1D, Rectangle 2D, Rectangle with hole, L-shape,
      Two-material plate, Layered slab, Box 3D, axisymmetric
      cylinder, hollow cylinder, solid 3D cylinder, polar
      cylindrical shell) cover most academic and prototyping cases.
      Pick one, dial in the parameters, and a 3D preview updates
      as you type — no exporting to a separate mesh file format
      and re-importing.
    </p>
    <p>
      The 2D paint canvas lets you assign material regions and
      boundary IDs visually before the mesh is ever generated.
      Five curated geometry-mapping modes — piecewise linear,
      circular arc, line-to-circle, line-to-sinusoid, and
      polygonal-angle — let you bend the otherwise-rectangular
      subcell boundaries into curves so the meshed result follows
      the actual shape of your domain.
    </p>
    <p>
      Three dedicated wizards handle the structurally bigger
      operations: a <strong>Boundary Layer</strong> wizard for
      adding inflated boundary-aligned layers (typical for CFD
      wall treatment), an <strong>Extrusion</strong> wizard that
      lifts a 2D mesh into 3D as a stack of layers, and a
      <strong>Revolution</strong> wizard that sweeps a 2D section
      around an axis to produce an axisymmetric or full 3D solid.
      Each wizard ships with a live preview pane so you can
      sanity-check divisions and limits before committing.
    </p>
  </div>
</div>

<div class="row mt-3">
  <div class="col-md-12">
    <img src="{{ '/assets/img/femstudio/mesh_generator.png' | relative_url }}"
         alt="Mesh generator across four geometry templates: a rectangle with a hole, an axisymmetric solid cylinder, a hollow polar cylindrical shell, and a curved domain."
         class="img-fluid rounded z-depth-1">
    <p class="text-center mt-2">
      <small class="text-muted">
        Four mesh-generator examples: rectangle with a hole, axisymmetric
        solid cylinder, hollow polar cylindrical shell, and a curved domain
        produced via geometry mappings.  The 2D paint canvas, parametric
        controls, and live 3D preview update together as you edit.
      </small>
    </p>
  </div>
</div>

<!-- ===========================================================
     Mesh import + ElmerGrid operations through GUI
     =========================================================== -->
<div class="row mt-5">
  <div class="col-md-12">
    <h2>Bring your own mesh, or clean up an existing one</h2>
    <p>
      External meshes import directly through <strong>File →
      Import Mesh…</strong>.  Supported formats are Gmsh
      (<code>.msh</code>), Universal / Salome (<code>.unv</code>),
      ElmerGrid native (<code>.grd</code>), and ANSYS / Abaqus
      input decks (<code>.ans</code>, <code>.inp</code>).
      ElmerStudio shells out to ElmerGrid behind the scenes with
      the right format codes, so you don't have to look up
      <code>-input 14</code> vs. <code>-input 17</code> ever
      again.  Existing Elmer mesh directories load by pointing
      at the folder.
    </p>
    <p>
      Once a mesh is loaded, the Mesh menu exposes the operations
      you'd otherwise run by hand from the ElmerGrid command line:
    </p>
    <ul>
      <li>
        <strong>Mesh → Cleanup</strong>: Auto Clean, Merge Nodes,
        Remove Unused Nodes — the "fix actual mesh defects"
        operations.  Each is a form-based dialog that runs
        ElmerGrid in place and reloads the result.
      </li>
      <li>
        <strong>Mesh → Transform</strong>: Scale, Translate, Rotate,
        Centralize.
      </li>
      <li>
        <strong>Mesh → Element Order</strong>: bump linear elements
        up to quadratic, or reduce a quadratic mesh back down.
      </li>
      <li>
        <strong>Mesh → Clone &amp; Mirror</strong>: replicate a
        mesh in space or reflect it across an axis.
      </li>
      <li>
        <strong>Mesh → Refine Mesh</strong>: uniform refinement
        that splits each element (triangle → 4, tetrahedron → 8).
      </li>
    </ul>
    <p>
      For inspecting what you've got, the same menu has
      <strong>Mesh Quality</strong> (toggles a colour overlay
      shading elements by quality metric so distorted regions
      jump out) and <strong>Mesh Statistics…</strong> (a popup
      with element counts, types, bounding box, and per-axis
      extents).  Display options — boundary colouring by ID,
      wireframe, ground grid, origin axes, clip plane, slice
      plane — all live in the same menu and persist across runs.
    </p>
  </div>
</div>

<!-- ===========================================================
     M9 / M9b / M9c / M9d — Bodies & Boundaries submenu — NEW in 0.7.0
     =========================================================== -->
<div class="row mt-5">
  <div class="col-md-12">
    <h2>Edit body and boundary IDs without re-meshing <small class="text-muted">(new in 0.7.0)</small></h2>
    <p>
      An import-time mesh almost never lands with the body and
      boundary IDs grouped the way your physics actually wants
      them.  Three recurring frustrations come up on the Elmer
      forum:
    </p>
    <ul>
      <li>
        <strong>"How do I group several boundary IDs under one
        Boundary Condition?"</strong>  The classic workaround is
        the `Type` column in <code>mesh.boundary</code>, but
        finding it from the GUI was historically opaque.
      </li>
      <li>
        <strong>"My plate has a hole and the outer and inner
        perimeters got bundled under one BC index."</strong>
        ElmerGrid's side-code system is direction-aware but not
        topology-aware, so non-simply-connected domains lump
        topologically-disconnected boundaries together.
      </li>
      <li>
        <strong>"I have multiple physical volumes but Elmer sees
        one body."</strong>  Cloning a
        mesh, mirroring it, or importing from a CAD tool that
        doesn't separate physical volumes at export all produce
        meshes where one ID covers multiple disconnected pieces.
      </li>
    </ul>
    <p>
      The new <strong>Mesh → Bodies &amp; Boundaries</strong>
      submenu collects fifteen operations that solve these
      directly, post-import, no re-meshing required.  Four
      categories:
    </p>
    <ul>
      <li>
        <strong>Merge</strong> ranges of body or boundary IDs into
        one (<strong>Merge Body Range…</strong> /
        <strong>Merge Boundary Range…</strong>).  Wraps the
        relevant ElmerGrid command-line flags but with a dialog
        that shows you the IDs and their element counts side-by-
        side, so you pick visually instead of trying to remember
        what's where.  This is the direct fix for the "group
        several BCs under one Boundary Condition" question.
      </li>
      <li>
        <strong>Add</strong> a new boundary at the interface
        between two bodies (<strong>Add Interface Boundary…</strong>)
        or at the intersection of two existing boundaries
        (<strong>Add Intersection Boundary…</strong>).  Useful
        for thermal contact, fluid-solid coupling, line loads,
        and edge constraints added without revisiting the
        mesher.
      </li>
      <li>
        <strong>Split a boundary</strong> four different ways.
        The headline is the one-click <strong>Split All
        Disconnected Boundaries</strong> — no dialog, no
        parameters: walks every boundary, finds the ones with
        more than one connected component, and splits each into
        fresh IDs.  For the canonical plate-with-hole import,
        that's one click and the outer and inner perimeters
        become separate BCs.  Per-boundary variants
        (<strong>Split Boundary by Components</strong>,
        <strong>by Feature Angle</strong>, <strong>by
        Plane</strong>, <strong>by Coordinate</strong>) handle
        more targeted cases — Feature Angle is the right tool
        for splitting a cube that imported with all six faces
        under one BC into per-face BCs (at the 30° default
        threshold, the cube's 90° corners register as feature
        edges and the six faces split cleanly).
      </li>
      <li>
        <strong>Split a body</strong> by connected components
        (<strong>Split Body by Components…</strong>), or
        extract a subset of bodies into a new self-contained
        mesh (<strong>Extract Bodies…</strong>).  Split Body
        closes the "after Clone, everything is one body" gap
        directly.  It offers face-adjacency (default — safer)
        or node-adjacency (more permissive — useful when
        touching-but-not-merged geometry should count as one
        body).
      </li>
    </ul>
    <p>
      Every dialog shows a live ElmerGrid command preview at the
      bottom, so you see the exact invocation that's about to run
      before clicking OK — handy for learning the underlying
      ElmerGrid flags or scripting the same operation later.
      ID dropdowns are labelled with element counts
      (<em>"3&nbsp;(1,842 elements)"</em>) so you pick the right
      one visually instead of by memory.  All operations write
      to a new output directory (suffixed with what was done,
      e.g. <code>_split_disconnected</code>) and leave the
      source mesh untouched.
    </p>
    <p>
      The five splitting operations are implemented in pure
      Python, directly editing <code>mesh.boundary</code> /
      <code>mesh.elements</code> — ElmerGrid has no command for
      splitting one boundary into many, so these fill a real
      gap rather than wrap an existing flag.  The body-component
      splitter and the topology-aware boundary splitter walk the
      element-adjacency graph properly, so they handle the
      pathological cases (a body that's "two cubes touching only
      at an edge", a boundary that's actually a Möbius strip)
      that simpler approaches would fail on.
    </p>
  </div>
</div>


<!-- ===========================================================
     Assembly tab — NEW in 0.7.0
     =========================================================== -->
<div class="row mt-5">
  <div class="col-md-12">
    <h2>Combine multiple meshes into one assembly <small class="text-muted">(new in 0.7.0)</small></h2>
    <p>
      Some problems need more than one mesh.  A rotor next to a
      stator, a fuel rod inside a coolant channel, a pipe with an
      embedded sensor — each part probably came from a different
      source (one from your CAD export via Gmsh, another from an
      Elmer <code>.grd</code> script, a third from a Salome
      <code>.unv</code>), and historically the only way to join
      them was a multi-step ElmerGrid dance: write a glue file by
      hand, get the offset numbering right, hope the boundary IDs
      didn't collide, hope you remembered the mortar BC keywords
      correctly when you got to the SIF.  I lost an afternoon to
      exactly this when a colleague asked for help with a
      rotor-stator coupling, and the result is the Assembly tab.
    </p>
    <p>
      The workflow is now: add each source mesh as a
      <strong>Part</strong>, give it a position and orientation,
      preview the whole thing in 3D, and bake.  Three bake modes
      cover the common cases:
    </p>
    <ul>
      <li>
        <strong>Concatenate</strong> — drop the parts side-by-side
        with offset IDs.  Useful when the parts are spatially
        separate (e.g., two PCBs on a board) or when you want to
        author the inter-part interfaces yourself in the SIF.
      </li>
      <li>
        <strong>Weld</strong> — collapse coincident nodes across
        parts within a tolerance, so meshes that nominally touch
        but came from different sources become a single
        watertight mesh.
      </li>
      <li>
        <strong>Interface</strong> — keep both sides of every
        detected mating face distinct (each gets its own boundary
        ID) and emit a <code>mesh.mortars</code> sidecar listing
        the pairs.  This is the right mode for sliding contacts,
        non-conforming meshes, and rotor-stator setups.
      </li>
    </ul>
    <p>
      Per-Part topology operations let you tidy each mesh before
      the bake without ever modifying the source files on disk:
      merge boundaries that the source mesher split unhelpfully,
      delete an interior interface that isn't really a boundary,
      extract only the bodies you want.  Per-Part naming lets you
      label each body and boundary ID with something
      human-readable — <code>rotor_slide</code>, <code>stator_outer</code>,
      <code>core</code> — and those labels flow through to the
      baked <code>mesh.names</code>, the integrated Mesh tab's
      region selectors, and any SIF you write afterward.
    </p>
    <p>
      For Interface-mode bakes, the <strong>Generate Mortar
      SIF…</strong> action turns the detected mortar pairs into
      ready-to-paste <code>Boundary Condition</code> blocks with
      proper <code>Mortar BC = Integer N</code> linkage, a
      projector kind auto-detected from the interface geometry
      (Rotational for cylindrical sliding boundaries, Radial for
      parallel periodic planes, Level Projector Generic as a
      universal fallback), and a <code>Galerkin Projector =
      Logical True</code> on every master block.  The most common
      Elmer-forum support thread, eliminated.
    </p>
    <p>
      A <strong>Pick</strong> mode in the toolbar handles the
      flip side: when you've imported a mesh and don't know which
      numeric boundary tag corresponds to which physical face,
      switch Pick to <em>Boundary</em>, click a face in the 3D
      preview, and the status bar tells you the ID (and how many
      cells the group contains).  The whole boundary group lights
      up translucent orange so you see what you've actually
      identified, not just one facet.  Body picks work the same
      way.  Once you know which IDs are which, label them in
      Edit Names and the rest of the workflow uses your labels.
    </p>
  </div>
</div>

<div class="row mt-3">
  <div class="col-md-12">
    <img src="{{ '/assets/img/femstudio/shaft_sleeve_demo.png' | relative_url }}"
         alt="Assembly tab showing two coaxial cylinders — a red inner shaft and a blue outer annular sleeve — placed at the origin. The right panel lists the selected shaft Part's IDs (Bodies 1, 2 and Boundaries 3-6), placement transforms, and Bake options."
         class="img-fluid rounded z-depth-1">
    <p class="text-center mt-2">
      <small class="text-muted">
        Two coaxial cylinders composed in the Assembly tab.  The
        red inner shaft (two material subdomains — core and ring)
        is selected; the blue outer sleeve sits coaxially around
        it.  Their mating surface at <code>r=1</code> becomes a
        rotational mortar pair after an Interface-mode bake, and
        Generate Mortar SIF emits the matching
        <code>Boundary Condition</code> blocks.
      </small>
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
      SIF into <strong>Setup</strong> (Header, Constants, Simulation),
      <strong>Physics</strong> (Solver, Equation),
      <strong>Domain</strong> (Body, Material, Body Force, Initial
      Condition, Component), and <strong>Boundaries</strong>
      (Boundary Condition).  Click any entry to jump to it; right-click
      to delete, duplicate, or add a new section in place.
      Validation badges flag problems inline so issues find you,
      not the other way around.
    </p>
    <p>
      The middle pane is a syntax-highlighted SIF editor with a
      line-number gutter, current-line highlight, monospace zoom
      (Ctrl + / Ctrl − / Ctrl 0), Find/Replace bar (Ctrl+F /
      Ctrl+H), and a comment-toggle shortcut (Ctrl + /).  It
      round-trips losslessly with the form-based dock panels —
      edit either side and the other stays in sync.  Dedicated
      docks handle the common authoring surfaces: a
      <strong>Simulation Setup</strong> dock for the Simulation
      and Constants sections, a <strong>Solver Setup</strong> dock
      that picks a solver from the bundled definitions and
      auto-renders its parameters as a form (with Equation, Solver,
      Body Force, and Initial Condition tabs), a
      <strong>Material Library</strong> dock with a category filter
      and a properties preview that injects materials with units
      handled, and a <strong>Boundary Conditions</strong> form for
      authoring BCs without touching the SIF.
    </p>
    <p>
      <strong>SIF keyword autocomplete <small class="text-muted">(new in 0.7.0)</small></strong>
      &mdash; two characters into a keyword inside any known
      section (Solver, Material, Body Force, Boundary Condition,
      …) and a popup appears with the keywords valid for
      <em>that section</em>, ranked by exact-prefix match first
      then substring match.  Tooltips show what each keyword does
      (the <code>Whatis</code> text from the bundled EDFs for
      EDF-sourced keywords, hand-written descriptions for the
      Header / Simulation / Constants / Body sections).
      Combo-valued keywords also complete to their allowed values
      once the user is past the <code>=</code> &mdash; type
      <code>Convection = </code> in a Solver block and the popup
      offers <code>None</code> / <code>Constant</code> /
      <code>Computed</code>.  Top-level (between sections) the
      popup suggests section kinds.  Esc dismisses; Enter / Tab
      accepts.  Imported ElmerGUI EDF files contribute their
      keywords to the popup the same way the bundled ones do.
    </p>
    <p>
      The right pane is an integrated VTU results viewer with field
      selection, eight built-in colormaps (viridis, plasma, coolwarm,
      RdYlBu, jet, turbo, inferno, magma), warp-by-vector with a
      scale factor, and a clip plane.  For transient runs, a
      timestep slider beneath the viewport scrubs through the
      sequence while preserving your scalar selection.  When you
      do need ParaView's full power, one click opens the current
      results in it.  The "Open in ParaView" path also transparently
      re-encodes the VTU on the way out to dodge the libexpat
      2.6.0–2.6.2 bug that breaks ParaView on Ubuntu 24.04+ — so
      it just works on a stock install with no patching.
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
      times or writing shell scripts that templated the SIF.
      ElmerStudio gives parameter sweeps a dedicated panel in the
      main window, no manual file editing required.  Pick a
      keyword from any section in your SIF, declare the values
      you want to sweep over, and ElmerStudio handles the rest.
    </p>
    <p>
      Two sweep modes ship out of the box.  In
      <strong>Cartesian</strong> mode, ElmerStudio runs every
      combination of every parameter — full factorial, useful when
      you want to map the response surface across two or more
      independent dimensions.  In <strong>Zipped</strong> mode,
      parameters advance together in lockstep, so you get N
      pre-paired configurations rather than N×M combinations —
      useful when you want to vary several keywords together along
      a curve in parameter space.
    </p>
    <p>
      Each sweep step runs in its own subdirectory with its own
      generated SIF, runs the solver, and surfaces results in a
      live status table as the sweep progresses — values, exit
      code, wall time, and any error per step.  You can abort the
      whole sweep, skip just the current step, or watch the
      convergence plot live for the active run.  When the sweep
      finishes, the results viewer's slider scrubs through the
      output VTUs from each step while preserving your chosen
      scalar field, and a separate summary chart plots a chosen
      output metric (max / min / mean) against the swept parameter
      — line chart for one parameter, heatmap for two on a
      Cartesian grid.  The structured directory tree of every
      run is also yours to post-process or feed into your own
      analysis script.
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
        runs.  Per-solver residuals stream into a chart on a log
        y-axis in real time, so you see divergence the moment it
        happens instead of staring at terminal output.
      </li>
      <li>
        <strong>A 35-entry materials library</strong> — air,
        nitrogen, argon, water, common metals, polymers, and the
        usual suspects — searchable by category with units handled
        for you.  One click injects the material properties into
        the active <code>Material</code> section.
      </li>
      <li>
        <strong>Pre-flight validation</strong> before each run.
        Verifies a mesh is loaded, at least one solver has a
        Procedure, every Equation has Active Solvers, every
        Boundary Condition has Target Boundaries assigned, and
        every Body references existing Material / Equation / Body
        Force / Initial Condition sections.  Catches the Elmer
        gotcha where two Body sections claiming the same mesh ID
        silently honor only the first.
      </li>
      <li>
        <strong>Live solver output</strong> with a warning banner.
        ElmerSolver's stdout / stderr stream into a coloured pane
        as the run progresses.  Lines matching known warning
        patterns are also collected separately and surfaced in a
        summary banner at the top, so warnings buried thousands
        of lines deep in the log don't go unseen.
      </li>
      <li>
        <strong>Eight bundled solver definitions</strong> covering
        common physics (heat equation, linear elasticity,
        Navier-Stokes, electrostatics, Helmholtz,
        advection-diffusion, static current, result output).
        Each renders as a structured form with the right keywords
        for that solver.  Custom ElmerGUI-style EDF files import
        via <strong>File → Import Solver Definition…</strong>.
      </li>
      <li>
        <strong>Parallel solver support</strong> through a settings
        dialog — pick the number of MPI processes and ElmerStudio
        invokes <code>ElmerSolver_mpi</code> with the right
        partitioning.  Settings persist across sessions.
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
        <strong>It isn't the solver.</strong>  ElmerStudio is a
        frontend.  You need ElmerSolver itself installed and on
        your PATH for simulations to run.  Get it from
        <a href="http://www.elmerfem.org/blog/binaries/" target="_blank" rel="noopener">elmerfem.org</a>
        or your distribution's package manager.
      </li>
      <li>
        <strong>It isn't a CAD tool.</strong>  The mesh generator
        covers parametric primitives well but doesn't do general
        CAD import.  For complex geometries, mesh externally with
        Gmsh, Salome, or your CAD tool's mesh export, then load
        the resulting <code>.msh</code> / <code>.unv</code> /
        <code>.grd</code> / <code>.ans</code> / <code>.inp</code>
        file into ElmerStudio.
      </li>
      <li>
        <strong>It isn't every Elmer feature.</strong>  Elmer's
        solver has decades of accumulated capability.  ElmerStudio
        ships eight solver definition files covering the most
        common physics; the rest is still accessible by editing
        the SIF directly in the integrated text view.  If a feature
        you need would benefit from a structured form,
        <a href="https://github.com/FEMStudio/femstudio-releases/issues" target="_blank" rel="noopener">file an issue</a>.
      </li>
      <li>
        <strong>It isn't open source.</strong>  Free as in beer,
        not as in speech.  See the license for what's allowed and
        what isn't.  This may change in the future.
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
      ElmerStudio version (Help → About in the app, or run with
      <code>--version</code>), and what you were doing when it
      happened.
    </p>
    <p>
      I read everything but reply on a rolling basis — this is a side
      effort, not a full-time gig.
    </p>
  </div>
</div>