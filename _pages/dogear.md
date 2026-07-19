---
layout: page
permalink: /dogear/
title: Dogear
nav: false
description: An Obsidian plugin that turns your vault into a reading tracker- Goodreads import, progress by page, percentage or listening time, a cover grid, and reading statistics, all in plain Markdown.
---

<div class="row mt-4">
  <div class="col-md-12">
    <p class="lead">
      Dogear is a community plugin for
      <a href="https://obsidian.md/" target="_blank" rel="noopener">Obsidian</a>
      that keeps your reading life in your own notes.  It brings your library
      across from Goodreads, stores one Markdown note per book, and logs your
      progress by page, by percentage, by time elapsed, or by time remaining.
      It grew out of the closure of the Goodreads API and the slow decay of the
      plugins that once depended on it: a decade of reading history should not
      sit behind someone else's login.
    </p>
  </div>
</div>

<!-- =========================================================== Links -->
<div class="row text-center mt-4 mb-2">
  <div class="col-md-6 mb-3">
    <a href="obsidian://show-plugin?id=dogear"
       class="btn btn-primary btn-lg" style="min-width: 240px;">
      <i class="fas fa-book-open"></i>&nbsp;&nbsp;Open in Obsidian
    </a>
    <p class="mt-2"><small class="text-muted">Opens the plugin page in the app.</small></p>
  </div>
  <div class="col-md-6 mb-3">
    <a href="https://github.com/MasoudMiM/obsidian-dogear"
       class="btn btn-primary btn-lg" target="_blank" rel="noopener" style="min-width: 240px;">
      <i class="fab fa-github"></i>&nbsp;&nbsp;Source on GitHub
    </a>
    <p class="mt-2"><small class="text-muted">MIT licensed.</small></p>
  </div>
</div>

<!-- =========================================================== Hero -->
<div class="row mt-4">
  <div class="col-md-12 text-center">
    <img src="{{ '/assets/dogear/library.png' | relative_url }}"
         alt="The Dogear library view in Obsidian: a grid of book covers with shelf tabs across the top, a search box, and a sort control."
         class="img-fluid rounded z-depth-1">
    <p class="mt-2"><small class="text-muted">The library view: every book you have, filtered by shelf and searchable by title, author or series.</small></p>
  </div>
</div>

<!-- =========================================================== Data -->
<div class="row mt-4">
  <div class="col-md-12">
    <h2>One note per book</h2>
    <p>
      Nothing is hidden in a database.  Each book is a Markdown file with its
      metadata in the frontmatter and its reading log in the body, so the whole
      library is searchable, linkable, and backed up alongside everything else
      in the vault.  Anything you write yourself is left exactly as you wrote
      it, byte for byte, however many times the plugin rewrites the log above
      it.
    </p>
    <pre><code>---
dogear: 1
title: The Power Broker
authors:
  - Robert A. Caro
isbn13: "9780394480763"
pages: 1246
status: finished
rating: 5
---

## Reading log

### Read 1 &mdash; print &middot; 2024-01-04 &rarr; 2024-03-12

- 2024-01-14 &middot; page 210 &middot; 17%
- 2024-02-02 &middot; page 604 &middot; 48%

## Notes

Anything you write here is yours.
</code></pre>
    <p>
      If the plugin is uninstalled tomorrow, every book remains a perfectly
      readable note.  That is the point.
    </p>
  </div>
</div>

<!-- =========================================================== Import -->
<div class="row mt-4">
  <div class="col-md-12">
    <h2>Bringing a Goodreads library across</h2>
    <p>
      Export your library from Goodreads and import the CSV.  Shelves become
      statuses, ratings and read dates carry over, series names are lifted out
      of the titles where Goodreads hides them, and your reviews and private
      notes are preserved as prose.  Nothing is written to the vault until you
      have seen exactly what will be created, and any book can be deselected.
    </p>
    <p>
      Where the export is lossy, the importer says so rather than inventing
      history.  Goodreads records a single date even for a book you read four
      times, and no importer can recover the rest; Dogear reports the true count
      and admits the gap in the note.  A tracker that quietly fabricates your
      reading history is worse than one that tells you what it could not know.
    </p>
  </div>
</div>

<div class="row mt-2">
  <div class="col-md-12 text-center">
    <img src="{{ '/assets/dogear/import.png' | relative_url }}"
         alt="The Goodreads import screen: a count of books found, a panel of warnings about lossy fields, and a scrollable list of books with individual checkboxes."
         class="img-fluid rounded z-depth-1">
    <p class="mt-2"><small class="text-muted">The import preview, with the losses stated before a single file is written.</small></p>
  </div>
</div>

<!-- =========================================================== Progress -->
<div class="row mt-4">
  <div class="col-md-12">
    <h2>Progress, however you actually read</h2>
    <p>
      Log a position by page, by percentage, by time elapsed, or by
      <em>time remaining</em>.  That last one matters: audiobook apps report how
      much is left rather than how far you have come, and no other tracker lets
      you record it that way.  Everything is normalised to a fraction of the
      book, so progress stays comparable when you switch between a paperback and
      an audiobook halfway through.
    </p>
    <p>
      Rereads are simply a second session, so a book can carry as many passes as
      you like without overwriting the first.  Abandoning a book records where
      you stopped and why, because that is information too.
    </p>
  </div>
</div>

<div class="row mt-2">
  <div class="col-md-12 text-center">
    <img src="{{ '/assets/dogear/panel.png' | relative_url }}"
         alt="The Dogear panel inside a book note: shelf buttons, a progress bar, a format picker, a unit selector, and controls to log progress or finish the book."
         class="img-fluid rounded z-depth-1">
    <p class="mt-2"><small class="text-muted">The panel inside each note, which changes with the state of the book rather than showing every control at once.</small></p>
  </div>
</div>

<!-- =========================================================== Stats -->
<div class="row mt-4">
  <div class="col-md-12">
    <h2>Statistics that stay honest</h2>
    <p>
      Books and pages finished, how you rated them, how long they were, and
      which authors you came back to, by year or across everything.  Audiobooks
      are counted in hours rather than pages, so listening is not quietly left
      out of your totals.
    </p>
    <p>
      It is deliberately small.  Every figure works from a finish date, a rating
      and a page count, because that is all an imported library knows: Goodreads
      exports no start dates, so reading pace and streaks cannot be computed for
      the great majority of anyone's books.  A page of empty charts helps
      nobody, so there is no mood tracking, no genre wheel, and no streaks.
    </p>
  </div>
</div>

<div class="row mt-2">
  <div class="col-md-12 text-center">
    <img src="{{ '/assets/dogear/stats.png' | relative_url }}"
         alt="The Dogear statistics view: headline figures for books and pages, a bar chart of books finished each month, a rating distribution, and a breakdown of book lengths."
         class="img-fluid rounded z-depth-1">
    <p class="mt-2"><small class="text-muted">Reading statistics for a selected year, drawn from finish dates, ratings and page counts alone.</small></p>
  </div>
</div>

<!-- =========================================================== Features -->
<div class="row mt-4">
  <div class="col-md-12">
    <h2>What's inside</h2>
    <ul>
      <li>
        <strong>Goodreads import</strong> with a full preview, per-book
        selection, duplicate detection, and honest warnings wherever the export
        loses information.
      </li>
      <li>
        <strong>A library view</strong> with a cover grid, shelf filters,
        accent-insensitive search across titles, authors and series, and sorting
        by recent activity, title, author or rating.
      </li>
      <li>
        <strong>Four ways to record a position</strong>: page, percentage, time
        elapsed and time remaining, all normalised so mixed-format reads stay
        coherent.
      </li>
      <li>
        <strong>Metadata from four catalogues</strong>: Open Library, Google
        Books, the Library of Congress and the Internet Archive, tried in order
        with automatic fallback and client-side rate limiting, so their limits
        are respected rather than discovered.
      </li>
      <li>
        <strong>Covers that survive</strong>: fetched from Open Library by ISBN
        at no request cost, downloadable into the vault so they work offline, or
        set by hand from any image you have, which is the only option for the
        many books no catalogue has artwork for.
      </li>
      <li>
        <strong>Manual entry that always works</strong>.  Looking a book up is
        enrichment, never a prerequisite, so a book can be added with no network
        at all.
      </li>
      <li>
        <strong>A tested core</strong>: the logic layer imports neither the DOM
        nor the Obsidian API and is covered by more than 1,100 tests, including
        fuzzing that checks your own writing survives repeated rewrites
        untouched.
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
      Browse</strong>, search for <em>Dogear</em>, install, and enable it.  Open
      your library from the ribbon icon, or run <em>Import from Goodreads</em>
      from the command palette to bring an existing library across.  The plugin
      is free and open source under the MIT license, and works on desktop and
      mobile.
    </p>
    <p>
      Found a bug or want another feature?  The
      <a href="https://github.com/MasoudMiM/obsidian-dogear/issues" target="_blank" rel="noopener">issue tracker</a>
      is open.
    </p>
  </div>
</div>
