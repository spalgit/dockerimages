/* Jump to a compound pair in the matrix.
 *
 * The header has two boxes: a row (POI) compound and a column (E3) compound.
 * Enter or "Go" scrolls the matrix box so the matching cell sits just inside
 * the frozen header rows and structure columns, and highlights it. With only
 * one box filled it jumps to that row or that column.
 *
 * Rows carry data-poi and E3 headers / cells carry data-e3 (render_matrix in
 * sar_app_cdd.py), so all of this happens in the browser — no callback, no
 * redraw. The suggestion lists are rebuilt from whatever matrix is on screen. */
(function () {
  "use strict";

  var FLASH_MS = 2500;

  function $(id) { return document.getElementById(id); }

  function ensureList(id) {
    var list = $(id);
    if (!list) {
      list = document.createElement("datalist");
      list.id = id;
      document.body.appendChild(list);
    }
    return list;
  }

  function names(selector, attr) {
    var seen = {}, out = [];
    document.querySelectorAll(".sar-table " + selector).forEach(function (el) {
      var name = el.getAttribute(attr);
      if (name && !seen[name]) { seen[name] = true; out.push(name); }
    });
    return out;
  }

  var lastSignature = "";
  function refreshLists() {
    var rows = names("tr[data-poi]", "data-poi");
    var cols = names("th[data-e3]", "data-e3");
    var signature = rows.join("|") + "#" + cols.join("|");
    if (signature === lastSignature) return;
    lastSignature = signature;
    [["find-poi-list", rows], ["find-e3-list", cols]].forEach(function (pair) {
      var list = ensureList(pair[0]);
      list.innerHTML = "";
      pair[1].forEach(function (name) {
        var opt = document.createElement("option");
        opt.value = name;
        list.appendChild(opt);
      });
    });
  }

  // Exact (case-insensitive) match first; otherwise a unique partial match, so
  // "645" finds A645 as long as nothing else contains 645.
  function resolve(query, available) {
    var q = query.trim().toLowerCase();
    if (!q) return { name: null };
    var exact = available.filter(function (n) { return n.toLowerCase() === q; });
    if (exact.length) return { name: exact[0] };
    var partial = available.filter(function (n) { return n.toLowerCase().indexOf(q) !== -1; });
    if (partial.length === 1) return { name: partial[0] };
    if (partial.length > 1) {
      return { error: "“" + query.trim() + "” matches " + partial.length + ": " +
               partial.slice(0, 6).join(", ") + (partial.length > 6 ? ", …" : "") };
    }
    return { error: "“" + query.trim() + "” is not in this matrix" };
  }

  function esc(value) { return value.replace(/\\/g, "\\\\").replace(/"/g, '\\"'); }

  function clearHighlights() {
    document.querySelectorAll(".sar-table .find-hit, .sar-table .find-line")
      .forEach(function (el) { el.classList.remove("find-hit", "find-line", "find-flash"); });
  }

  function mark(elements, cls) {
    elements.forEach(function (el) {
      el.classList.add(cls, "find-flash");
      window.setTimeout(function () { el.classList.remove("find-flash"); }, FLASH_MS);
    });
  }

  // Scroll the box so `target` lands just past the sticky header / columns.
  function scrollToCell(box, target, horizontal, vertical) {
    var boxRect = box.getBoundingClientRect();
    var rect = target.getBoundingClientRect();
    if (horizontal) {
      var frozen = box.querySelector("tbody .poi-name") || box.querySelector(".poi-name");
      var left = frozen ? frozen.getBoundingClientRect().right : boxRect.left;
      box.scrollLeft += rect.left - left - 8;
    }
    if (vertical) {
      // The <thead> itself scrolls away; only its sticky cells stay put.
      var top = boxRect.top;
      box.querySelectorAll("thead th").forEach(function (th) {
        top = Math.max(top, th.getBoundingClientRect().bottom);
      });
      box.scrollTop += rect.top - top - 4;
    }
  }

  function setMessage(text) {
    var msg = $("find-msg");
    if (msg) msg.textContent = text || "";
  }

  function find() {
    var box = document.querySelector(".table-scroll");
    var poiInput = $("find-poi"), e3Input = $("find-e3");
    if (!box || !poiInput || !e3Input) { setMessage("No matrix on screen"); return; }
    refreshLists();

    var poi = resolve(poiInput.value, names("tr[data-poi]", "data-poi"));
    var e3 = resolve(e3Input.value, names("th[data-e3]", "data-e3"));
    var errors = [poi.error, e3.error].filter(Boolean);
    if (errors.length) { setMessage(errors.join(" · ")); return; }
    if (!poi.name && !e3.name) { setMessage("Type a row and/or column compound"); return; }

    clearHighlights();
    var row = poi.name ? box.querySelector('tr[data-poi="' + esc(poi.name) + '"]') : null;
    var head = e3.name ? box.querySelector('th[data-e3="' + esc(e3.name) + '"]') : null;
    if (poi.name) poiInput.value = poi.name;
    if (e3.name) e3Input.value = e3.name;

    if (row && head) {
      var cells = Array.prototype.slice.call(
        row.querySelectorAll('td[data-e3="' + esc(e3.name) + '"]'));
      mark([head, row.querySelector(".poi-name")].filter(Boolean), "find-line");
      mark(cells, "find-hit");
      scrollToCell(box, cells[0] || head, true, true);
      var measured = cells.some(function (c) { return c.textContent.trim() !== ""; });
      setMessage(poi.name + " × " + e3.name + (measured ? "" : " — no data for this pair"));
    } else if (row) {
      mark([row.querySelector(".poi-name")].filter(Boolean), "find-line");
      mark(Array.prototype.slice.call(row.querySelectorAll("td")), "find-hit");
      scrollToCell(box, row, false, true);
      setMessage("Row " + poi.name);
    } else if (head) {
      mark([head], "find-line");
      mark(Array.prototype.slice.call(
        box.querySelectorAll('td[data-e3="' + esc(e3.name) + '"]')), "find-hit");
      scrollToCell(box, head, true, false);
      setMessage("Column " + e3.name);
    }
  }

  document.addEventListener("click", function (event) {
    if (event.target.closest && event.target.closest("#find-go")) find();
  });
  document.addEventListener("keydown", function (event) {
    var id = event.target && event.target.id;
    if (event.key === "Enter" && (id === "find-poi" || id === "find-e3")) {
      event.preventDefault();
      find();
    }
  });
  // Keep the suggestions in step with the matrix Dash has drawn.
  document.addEventListener("focusin", function (event) {
    var id = event.target && event.target.id;
    if (id === "find-poi" || id === "find-e3") refreshLists();
  });
})();
