// Scroll-reveal: fire as soon as any part of the element enters the viewport,
// so tall sections still reveal on short viewports. Falls back to always-visible.
(function () {
  var items = document.querySelectorAll(".reveal");
  if (!("IntersectionObserver" in window)) {
    items.forEach(function (el) { el.classList.add("in"); });
    return;
  }
  var io = new IntersectionObserver(function (entries) {
    entries.forEach(function (entry) {
      if (entry.isIntersecting) {
        entry.target.classList.add("in");
        io.unobserve(entry.target);
      }
    });
  }, { threshold: 0, rootMargin: "0px 0px -5% 0px" });
  items.forEach(function (el) { io.observe(el); });
})();
