/* Keep the matrix's scrollbars on screen.
 *
 * The matrix scrolls inside its own box. With a fixed max-height that box ran
 * past the bottom of the window, so its horizontal scrollbar only came into
 * view after scrolling the page to the bottom. Here the box is sized to the
 * room left below it in the window, recomputed whenever the window is resized,
 * the controls above it wrap differently, or Dash redraws the matrix. */
(function () {
  "use strict";

  var BOTTOM_GAP = 16;    // breathing room under the box
  var MIN_HEIGHT = 320;   // on a very short window, scroll the page instead

  function fit() {
    var box = document.querySelector(".table-scroll");
    if (!box) return;
    // Measured from the top of the document, so the size does not change
    // with the page's own scroll position.
    var top = box.getBoundingClientRect().top + window.scrollY;
    var room = window.innerHeight - top - BOTTOM_GAP;
    box.style.maxHeight = Math.max(MIN_HEIGHT, room) + "px";
  }

  var pending = false;
  function schedule() {
    if (pending) return;
    pending = true;
    window.requestAnimationFrame(function () {
      pending = false;
      fit();
    });
  }

  window.addEventListener("resize", schedule);
  // Dash re-renders the matrix and the controls in place; watch for it.
  new MutationObserver(schedule).observe(document.body, { childList: true, subtree: true });
  if (window.ResizeObserver) {
    new ResizeObserver(schedule).observe(document.body);
  }
  schedule();
})();
