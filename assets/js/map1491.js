/* 1491 map widget for masoudmim.com — sidebar + Leaflet map.
   Requires window.PLACES_1491 (loaded from ../data/1491-data.js) and Leaflet. */
(function () {
  "use strict";

  var TYPE_COLORS = {
    city: "#d73027", country: "#66a16f", region: "#a6b96a", river: "#1f78b4",
    sea: "#4393c3", mountain: "#8c510a", island: "#e66101", lake: "#3288bd",
    settlement: "#9e0142", indigenous_territory: "#c26b4e", empire: "#7b3294",
    colony: "#ff7f00", archaeological_site: "#e6a700", museum: "#6a3d9a",
    park: "#00887c", road: "#999999", continent: "#bbbbbb", other: "#777777",
  };

  var PART_NAMES = {
    part_one: "Part One · Numbers from Nowhere",
    part_two: "Part Two · Very Old Bones",
    part_three: "Part Three · Landscape with Figures",
  };

  function isDivider(slug) { return /part_(one|two|three)/.test(slug); }

  function groupChapters(agg) {
    var groups = [{ label: "Front matter", slugs: [] }];
    var cur = groups[0];
    agg.chapters.forEach(function (ch) {
      var m = ch.slug.match(/part_(one|two|three)/);
      var n = parseInt(ch.slug, 10);
      if (m) {
        cur = { label: PART_NAMES["part_" + m[1]], slugs: [ch.slug] };
        groups.push(cur);
      } else if (n >= 15) {
        if (!cur.tail) {
          cur = { label: "Coda · Afterword · Appendices · Back matter", slugs: [], tail: true };
          groups.push(cur);
        }
        cur.slugs.push(ch.slug);
      } else {
        cur.slugs.push(ch.slug);
      }
    });
    return groups;
  }

  function typeCounts(entries) {
    var counts = {};
    entries.forEach(function (e) { counts[e.type] = (counts[e.type] || 0) + 1; });
    return counts;
  }

  function locationKey(e) {
    return Number(e.latitude).toFixed(3) + "," + Number(e.longitude).toFixed(3);
  }

  function groupSameSpot(entries) {
    var map = new Map();
    entries.forEach(function (e) {
      var key = e.slug + "|" + locationKey(e);
      if (!map.has(key)) {
        map.set(key, { latitude: e.latitude, longitude: e.longitude, slug: e.slug, entries: [] });
      }
      map.get(key).entries.push(e);
    });
    return Array.from(map.values());
  }

  function escapeHtml(s) {
    return String(s).replace(/[&<>"]/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c];
    });
  }

  function el(tag, cls) {
    var node = document.createElement(tag);
    if (cls) node.className = cls;
    return node;
  }

  function init() {
    var root = document.getElementById("m1491");
    var agg = window.PLACES_1491;
    if (!root || !agg || typeof L === "undefined") return;

    var state = { chapters: new Set(), types: new Set(), query: "" };
    var allGroups = [], allSlugs = [], visibleSlugs = [];

    /* ---- data prep ---- */
    var entries = [];
    agg.chapters.forEach(function (ch) {
      ch.locations.forEach(function (e) {
        entries.push(Object.assign({}, e, { slug: ch.slug }));
      });
    });
    allSlugs = agg.chapters.map(function (c) { return c.slug; });
    allGroups = groupSameSpot(entries);
    var maxCount = agg.chapters.reduce(function (m, c) { return Math.max(m, c.location_count); }, 1);

    /* ---- map ---- */
    var isDark = document.documentElement.getAttribute("data-theme") === "dark";
    var map = L.map("m1491-map", {
      worldCopyJump: true, renderer: L.canvas(),
      scrollWheelZoom: false, zoomControl: true,
    }).setView([16, -60], 3);
    var tiles = L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
      maxZoom: 19,
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
    }).addTo(map);
    var fellBack = false;
    tiles.on("tileerror", function () {
      if (fellBack) return;
      fellBack = true;
      map.removeLayer(tiles);
      L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}", {
        maxZoom: 18,
        attribution: "Tiles © Esri — Source: Esri, Maxar, Earthstar Geographics",
      }).addTo(map);
    });

    var layerGroup = L.layerGroup().addTo(map);

    /* ---- sidebar building ---- */
    var chaptersBox = root.querySelector(".m1491-chapters");
    var legendBox = root.querySelector(".m1491-legend");
    var statsEl = root.querySelector(".m1491-stats");
    var progressEl = root.querySelector(".m1491-progress");

    groupChapters(agg).forEach(function (group) {
      var block = el("details", "m1491-block");
      block.open = true;
      var head = el("summary", "m1491-group-label");
      var total = 0;
      group.slugs.forEach(function (s) {
        total += agg.chapters.find(function (c) { return c.slug === s; }).location_count;
      });
      head.innerHTML = "<span>" + group.label + "</span><span class='m1491-group-count'>" + total + "</span>";
      block.appendChild(head);
      var body = el("div", "m1491-group-body");
      block.appendChild(body);
      chaptersBox.appendChild(block);

      group.slugs.forEach(function (slug) {
        var ch = agg.chapters.find(function (c) { return c.slug === slug; });
        state.chapters.add(slug);
        if (isDivider(slug)) return;
        visibleSlugs.push(slug);
        var segs = Object.entries(typeCounts(ch.locations))
          .sort(function (a, b) { return b[1] - a[1]; })
          .map(function (tc) {
            var pct = ((tc[1] / ch.location_count) * 100).toFixed(1);
            var color = TYPE_COLORS[tc[0]] || TYPE_COLORS.other;
            return "<i style='width:" + pct + "%;background:" + color + "'></i>";
          }).join("");
        var row = el("label", "m1491-row");
        row.innerHTML = "<input type='checkbox' checked>" +
          "<span class='m1491-row-text'><span class='m1491-row-title'>" + escapeHtml(ch.chapter_title) + "</span>" +
          "<span class='m1491-bar-track'><span class='m1491-bar' style='width:" +
          ((ch.location_count / maxCount) * 100).toFixed(1) + "%'>" + segs + "</span></span></span>" +
          "<span class='m1491-count'>" + ch.location_count + "</span>";
        var input = row.querySelector("input");
        input.addEventListener("change", function () {
          if (input.checked) state.chapters.add(slug); else state.chapters.delete(slug);
          row.classList.toggle("off", !input.checked);
          render();
        });
        body.appendChild(row);
      });
    });

    var counts = typeCounts(entries);
    Object.entries(counts).sort(function (a, b) { return b[1] - a[1]; }).forEach(function (tc) {
      var type = tc[0];
      state.types.add(type);
      var chip = el("button", "m1491-chip");
      chip.type = "button";
      var color = TYPE_COLORS[type] || TYPE_COLORS.other;
      chip.innerHTML = "<span class='m1491-dot' style='background:" + color + "'></span>" +
        "<span>" + type.replace(/_/g, " ") + "</span><span class='m1491-count'>" + tc[1] + "</span>";
      chip.addEventListener("click", function () {
        if (state.types.has(type)) state.types.delete(type); else state.types.add(type);
        chip.classList.toggle("off", !state.types.has(type));
        render();
      });
      legendBox.appendChild(chip);
    });

    /* ---- rendering ---- */
    function popupHtml(g) {
      return g.entries.map(function (e) {
        var color = TYPE_COLORS[e.type] || TYPE_COLORS.other;
        return "<div class='m1491-pop-entry'><div class='m1491-pop-title'>" + escapeHtml(e.name) + "</div>" +
          "<div class='m1491-pop-meta'><span class='m1491-pop-chip' style='background:" + color + "'>" +
          e.type.replace(/_/g, " ") + "</span><span class='m1491-pop-chapter'>" + escapeHtml(e.chapter) + "</span></div>" +
          "<div class='m1491-pop-body'>" + escapeHtml(e.importance) + "</div></div>";
      }).join("");
    }

    function render() {
      layerGroup.clearLayers();
      var q = state.query.trim().toLowerCase();
      var searching = q.length > 0;
      var shown = 0, visible = [];
      allGroups.forEach(function (g) {
        if (!state.chapters.has(g.slug)) return;
        var kept = g.entries.filter(function (e) {
          if (!state.types.has(e.type)) return false;
          if (q && e.name.toLowerCase().indexOf(q) === -1) return false;
          return true;
        });
        if (!kept.length) return;
        shown += kept.length;
        visible.push(g);
        var marker = L.circleMarker([g.latitude, g.longitude], {
          radius: searching ? 9 : 6.5,
          color: "#2b2b2b", weight: searching ? 2 : 1.4,
          fillColor: TYPE_COLORS[kept[0].type] || TYPE_COLORS.other,
          fillOpacity: 0.88,
        });
        marker.bindPopup(popupHtml({ entries: kept }), { maxWidth: 360 });
        marker.bindTooltip(kept.map(function (e) { return e.name; }).join(" · "),
          { direction: "top", offset: [0, -8] });
        layerGroup.addLayer(marker);
      });
      statsEl.textContent = shown + " of " + entries.length + " locations shown";
      progressEl.textContent = visibleSlugs.filter(function (s) {
        return state.chapters.has(s);
      }).length + "/" + visibleSlugs.length;
    }

    function fit() {
      render();
      var bounds = [];
      allGroups.forEach(function (g) {
        if (!state.chapters.has(g.slug)) return;
        bounds.push([g.latitude, g.longitude]);
      });
      if (bounds.length) map.fitBounds(L.latLngBounds(bounds), { padding: [30, 30] });
    }

    /* ---- controls ---- */
    root.querySelector(".m1491-all").addEventListener("click", function () {
      state.chapters = new Set(allSlugs);
      chaptersBox.querySelectorAll("input").forEach(function (i) {
        i.checked = true;
        i.closest(".m1491-row").classList.remove("off");
      });
      render();
    });
    root.querySelector(".m1491-none").addEventListener("click", function () {
      state.chapters = new Set();
      chaptersBox.querySelectorAll("input").forEach(function (i) {
        i.checked = false;
        i.closest(".m1491-row").classList.add("off");
      });
      render();
    });
    root.querySelector(".m1491-fit").addEventListener("click", fit);
    root.querySelector(".m1491-search").addEventListener("input", function (ev) {
      state.query = ev.target.value;
      render();
    });
    root.querySelector(".m1491-all-ty").addEventListener("click", function () {
      Object.keys(counts).forEach(function (t) { state.types.add(t); });
      legendBox.querySelectorAll(".m1491-chip").forEach(function (c) { c.classList.remove("off"); });
      render();
    });
    root.querySelector(".m1491-none-ty").addEventListener("click", function () {
      state.types.clear();
      legendBox.querySelectorAll(".m1491-chip").forEach(function (c) { c.classList.add("off"); });
      render();
    });

    /* dark-mode tile filter, follows al-folio theme toggle */
    function applyFilter(dark) {
      var pane = document.querySelector("#m1491-map .leaflet-tile-pane");
      if (pane) pane.style.filter = dark
        ? "invert(1) hue-rotate(180deg) brightness(0.8) contrast(1.2)" : "none";
    }
    applyFilter(isDark);
    new MutationObserver(function () {
      var nowDark = document.documentElement.getAttribute("data-theme") === "dark";
      if (nowDark !== isDark) { isDark = nowDark; applyFilter(nowDark); }
    }).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });

    map.whenReady(function () { fit(); setTimeout(function () { map.invalidateSize(); }, 150); });
    render();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
