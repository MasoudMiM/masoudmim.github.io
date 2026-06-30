---
layout: page
permalink: /sketch-mechanisms/
title: Sketch Mechanisms
nav: false
description: An Obsidian plugin that renders live, hand-drawn animations of 2D mechanisms — linkages, gears, cams, pendulums, and springs — from a simple code block.
---

<div class="row mt-4">
  <div class="col-md-12">
    <p class="lead">
      Sketch Mechanisms is a community plugin for
      <a href="https://obsidian.md/" target="_blank" rel="noopener">Obsidian</a>
      that turns a small code block into a live, looping, hand-drawn animation
      of a classic 2D mechanism.  It grew out of wanting to drop a quick
      kinematic sketch into my notes — a four-bar linkage, a slider-crank, a
      pair of meshing gears — without leaving the editor or exporting a video.
      The strokes are drawn with the same engine
      <a href="https://excalidraw.com/" target="_blank" rel="noopener">Excalidraw</a>
      uses, so everything has a whiteboard, sketched-by-hand look, and the
      colours follow your Obsidian theme.
    </p>
  </div>
</div>

<!-- =========================================================== Links -->
<div class="row text-center mt-4 mb-2">
  <div class="col-md-6 mb-3">
    <a href="obsidian://show-plugin?id=sketch-mechanisms"
       class="btn btn-primary btn-lg" style="min-width: 240px;">
      <i class="fas fa-cube"></i>&nbsp;&nbsp;Open in Obsidian
    </a>
    <p class="mt-2"><small class="text-muted">Opens the plugin page in the app.</small></p>
  </div>
  <div class="col-md-6 mb-3">
    <a href="https://github.com/MasoudMiM/sketch-mechanisms"
       class="btn btn-primary btn-lg" target="_blank" rel="noopener" style="min-width: 240px;">
      <i class="fab fa-github"></i>&nbsp;&nbsp;Source on GitHub
    </a>
    <p class="mt-2"><small class="text-muted">MIT licensed.</small></p>
  </div>
</div>

<!-- =========================================================== Gallery -->
<div class="row mt-4">
  <div class="col-md-12">
    <h2>Six mechanisms, animated in your notes</h2>
    <p>
      Each block renders one mechanism, drawn frame-by-frame and looping
      cleanly.  Linkages trace the path of a highlighted point; the dynamic
      systems are integrated so the motion is physically faithful, not just
      decorative.
    </p>
  </div>
</div>

<div class="row mt-2">
  <div class="col-md-4 col-6 mb-4 text-center">
    <img src="{{ '/assets/sketch-mechanisms/fourbar.gif' | relative_url }}"
         alt="Animated four-bar linkage tracing its coupler curve" class="img-fluid rounded z-depth-1">
    <p class="mt-2"><small class="text-muted">Four-bar linkage</small></p>
  </div>
  <div class="col-md-4 col-6 mb-4 text-center">
    <img src="{{ '/assets/sketch-mechanisms/slidercrank.gif' | relative_url }}"
         alt="Animated slider-crank mechanism" class="img-fluid rounded z-depth-1">
    <p class="mt-2"><small class="text-muted">Slider-crank</small></p>
  </div>
  <div class="col-md-4 col-6 mb-4 text-center">
    <img src="{{ '/assets/sketch-mechanisms/gears.gif' | relative_url }}"
         alt="Animated meshing gear pair" class="img-fluid rounded z-depth-1">
    <p class="mt-2"><small class="text-muted">Gear pair</small></p>
  </div>
  <div class="col-md-4 col-6 mb-4 text-center">
    <img src="{{ '/assets/sketch-mechanisms/cam.gif' | relative_url }}"
         alt="Animated cam and follower" class="img-fluid rounded z-depth-1">
    <p class="mt-2"><small class="text-muted">Cam &amp; follower</small></p>
  </div>
  <div class="col-md-4 col-6 mb-4 text-center">
    <img src="{{ '/assets/sketch-mechanisms/pendulum.gif' | relative_url }}"
         alt="Animated pendulum" class="img-fluid rounded z-depth-1">
    <p class="mt-2"><small class="text-muted">Pendulum</small></p>
  </div>
  <div class="col-md-4 col-6 mb-4 text-center">
    <img src="{{ '/assets/sketch-mechanisms/spring.gif' | relative_url }}"
         alt="Animated spring-mass oscillator" class="img-fluid rounded z-depth-1">
    <p class="mt-2"><small class="text-muted">Spring-mass</small></p>
  </div>
</div>

<!-- =========================================================== Usage -->
<div class="row mt-4">
  <div class="col-md-12">
    <h2>How it works</h2>
    <p>
      Add a fenced code block with the language <code>mechanism</code> and
      describe what you want.  The plugin renders an animated SVG inline, in
      both reading view and Live Preview.  Click any drawing to play or pause.
    </p>
    <pre><code>```mechanism
type: fourbar
shimmer: 0.3
speed: 1
```</code></pre>
    <p>
      Every mechanism has its own parameters — link lengths, gear teeth, cam
      lift, pendulum angle — plus shared options for playback speed, line
      roughness, a <em>shimmer</em> dial that controls how much the strokes
      re-draw each frame (from rock-steady to lively), an accent colour, and
      whether to show the traced path.  Full option tables live in the
      <a href="https://github.com/MasoudMiM/sketch-mechanisms#usage" target="_blank" rel="noopener">README</a>.
    </p>
  </div>
</div>

<!-- =========================================================== Features -->
<div class="row mt-4">
  <div class="col-md-12">
    <h2>What's inside</h2>
    <ul>
      <li>
        <strong>Six mechanism types</strong> — four-bar linkage,
        slider-crank, external or internal/ring gear pair, cam with a
        follower, pendulum, and spring-mass oscillator.
      </li>
      <li>
        <strong>Genuinely hand-drawn</strong> — rendered with
        <a href="https://roughjs.com/" target="_blank" rel="noopener">rough.js</a>,
        the library behind Excalidraw, so it matches that sketched aesthetic
        rather than looking like clean CAD output.
      </li>
      <li>
        <strong>Physically faithful</strong> — linkage positions come from
        exact geometry; the pendulum is integrated with RK4 so large-angle
        swings are correct, and both the pendulum and spring support optional
        damping.
      </li>
      <li>
        <strong>Theme-aware and mobile-friendly</strong> — ink and paper
        colours follow your Obsidian theme (light or dark), it runs on the
        mobile app, and it respects your reduced-motion setting.
      </li>
      <li>
        <strong>Tweakable</strong> — cam dwell and motion profile, gear tooth
        counts, a movable traced point on the linkages, playback speed, and
        more, all from the code block or global settings.
      </li>
    </ul>
  </div>
</div>

<!-- =========================================================== Install -->
<div class="row mt-4">
  <div class="col-md-12">
    <h2>Install</h2>
    <p>
      In Obsidian, open <strong>Settings → Community plugins → Browse</strong>,
      search for <em>Sketch Mechanisms</em>, install, and enable it.  Then add
      a <code>mechanism</code> code block to any note.  The plugin is free and
      open source under the MIT license.
    </p>
    <p>
      Found a bug or want another mechanism?  The
      <a href="https://github.com/MasoudMiM/sketch-mechanisms/issues" target="_blank" rel="noopener">issue tracker</a>
      is open.
    </p>
  </div>
</div>
