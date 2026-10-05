/* Dose-response curve on hover.
 *
 * The matrix is an HTML table, so there is no Plotly hover layer to lean on.
 * Each POI x E3 curve is rendered server-side into #curve-gallery once; a cell
 * only carries data-curve="<key>". One delegated listener copies the matching
 * image into a single floating panel, so the cost is one panel, not one panel
 * per cell — and the listener survives every re-render of the matrix. */
(function () {
  "use strict";

  var panel = null;
  var active = null;
  var GAP = 16;          // clearance between the cursor and the panel

  function ensurePanel() {
    if (panel) return panel;
    panel = document.createElement("div");
    panel.className = "curve-pop";
    panel.setAttribute("role", "tooltip");
    document.body.appendChild(panel);
    return panel;
  }

  // Every gallery image for this pair (one drawn curve today), each with an
  // optional title and caption.
  function imagesFor(key) {
    var gallery = document.getElementById("curve-gallery");
    if (!gallery) return [];
    return Array.prototype.slice.call(
      gallery.querySelectorAll('img[data-key="' + key.replace(/"/g, '\\"') + '"]'));
  }

  function fill(images) {
    panel.textContent = "";
    var title = images[0].getAttribute("data-title");
    if (title) {
      var head = document.createElement("div");
      head.className = "curve-pop-title";
      head.textContent = title;
      panel.appendChild(head);
    }
    // Side by side, so a pair made as two batches still fits on screen.
    var row = document.createElement("div");
    row.className = "curve-pop-row";
    images.forEach(function (src) {
      var block = document.createElement("div");
      var caption = src.getAttribute("data-caption");
      if (caption) {
        var cap = document.createElement("div");
        cap.className = "curve-pop-caption";
        cap.textContent = caption;
        block.appendChild(cap);
      }
      var img = document.createElement("img");
      img.alt = "";
      img.src = src.getAttribute("src");
      block.appendChild(img);
      row.appendChild(block);
    });
    panel.appendChild(row);
  }

  function place(x, y) {
    if (!panel) return;
    var w = panel.offsetWidth || 360;
    var h = panel.offsetHeight || 252;
    var left = x + GAP;
    var top = y + GAP;
    if (left + w > window.innerWidth - 8) left = x - GAP - w;      // flip left
    if (top + h > window.innerHeight - 8) top = y - GAP - h;       // flip up
    panel.style.left = Math.max(8, left) + "px";
    panel.style.top = Math.max(8, top) + "px";
  }

  function show(cell, x, y) {
    var key = cell.getAttribute("data-curve");
    var images = imagesFor(key);
    if (!images.length) return;
    ensurePanel();
    if (active !== key) {
      fill(images);
      active = key;
    }
    panel.classList.add("is-open");
    place(x, y);
  }

  function hide() {
    active = null;
    if (panel) panel.classList.remove("is-open");
  }

  function cellAt(target) {
    return target && target.closest ? target.closest("[data-curve]") : null;
  }

  document.addEventListener("mouseover", function (ev) {
    var cell = cellAt(ev.target);
    if (cell) show(cell, ev.clientX, ev.clientY);
    else if (!ev.target.closest || !ev.target.closest(".curve-pop")) hide();
  });

  document.addEventListener("mousemove", function (ev) {
    if (active && cellAt(ev.target)) place(ev.clientX, ev.clientY);
  });

  // Keyboard parity: the first cell of each block is a tab stop.
  document.addEventListener("focusin", function (ev) {
    var cell = cellAt(ev.target);
    if (!cell) return hide();
    var box = cell.getBoundingClientRect();
    show(cell, box.right, box.bottom);
  });
  document.addEventListener("focusout", hide);

  document.addEventListener("keydown", function (ev) {
    if (ev.key === "Escape") hide();
  });
  // A panel pinned to viewport coordinates would drift away from its cell.
  window.addEventListener("scroll", hide, true);
})();
