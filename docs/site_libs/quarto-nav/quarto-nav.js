const headroomChanged = new CustomEvent("quarto-hrChanged", {
  detail: {},
  bubbles: true,
  cancelable: false,
  composed: false,
});

window.document.addEventListener("DOMContentLoaded", function () {
  let init = false;

  // Manage the back to top button, if one is present.
  let lastScrollTop = window.pageYOffset || document.documentElement.scrollTop;
  const scrollDownBuffer = 5;
  const scrollUpBuffer = 35;
  const btn = document.getElementById("quarto-back-to-top");
  const hideBackToTop = () => {
    btn.style.display = "none";
  };
  const showBackToTop = () => {
    btn.style.display = "inline-block";
  };
  if (btn) {
    window.document.addEventListener(
      "scroll",
      function () {
        const currentScrollTop =
          window.pageYOffset || document.documentElement.scrollTop;

        // Shows and hides the button 'intelligently' as the user scrolls
        if (currentScrollTop - scrollDownBuffer > lastScrollTop) {
          hideBackToTop();
          lastScrollTop = currentScrollTop <= 0 ? 0 : currentScrollTop;
        } else if (currentScrollTop < lastScrollTop - scrollUpBuffer) {
          showBackToTop();
          lastScrollTop = currentScrollTop <= 0 ? 0 : currentScrollTop;
        }

        // Show the button at the bottom, hides it at the top
        if (currentScrollTop <= 0) {
          hideBackToTop();
        } else if (
          window.innerHeight + currentScrollTop >=
          document.body.offsetHeight
        ) {
          showBackToTop();
        }
      },
      false
    );
  }

  function throttle(func, wait) {
    var timeout;
    return function () {
      const context = this;
      const args = arguments;
      const later = function () {
        clearTimeout(timeout);
        timeout = null;
        func.apply(context, args);
      };

      if (!timeout) {
        timeout = setTimeout(later, wait);
      }
    };
  }

  function headerOffset() {
    // Set an offset if there is are fixed top navbar
    const headerEl = window.document.querySelector("header.fixed-top");
    if (headerEl) {
      return headerEl.clientHeight;
    } else {
      return 0;
    }
  }

  function footerOffset() {
    const footerEl = window.document.querySelector("footer.footer");
    if (footerEl) {
      return footerEl.clientHeight;
    } else {
      return 0;
    }
  }

  function updateDocumentOffsetWithoutAnimation() {
    updateDocumentOffset(false);
  }

  function updateDocumentOffset(animated) {
    // set body offset
    const topOffset = headerOffset();
    const bodyOffset = topOffset + footerOffset();
    const bodyEl = window.document.body;
    bodyEl.setAttribute("data-bs-offset", topOffset);
    bodyEl.style.paddingTop = topOffset + "px";

    // deal with sidebar offsets
    const sidebars = window.document.querySelectorAll(
      ".sidebar, .headroom-target"
    );
    sidebars.forEach((sidebar) => {
      if (!animated) {
        sidebar.classList.add("notransition");
        // Remove the no transition class after the animation has time to complete
        setTimeout(function () {
          sidebar.classList.remove("notransition");
        }, 201);
      }

      if (window.Headroom && sidebar.classList.contains("sidebar-unpinned")) {
        sidebar.style.top = "0";
        sidebar.style.maxHeight = "100vh";
      } else {
        sidebar.style.top = topOffset + "px";
        sidebar.style.maxHeight = "calc(100vh - " + topOffset + "px)";
      }
    });

    // allow space for footer
    const mainContainer = window.document.querySelector(".quarto-container");
    if (mainContainer) {
      mainContainer.style.minHeight = "calc(100vh - " + bodyOffset + "px)";
    }

    // link offset
    let linkStyle = window.document.querySelector("#quarto-target-style");
    if (!linkStyle) {
      linkStyle = window.document.createElement("style");
      linkStyle.setAttribute("id", "quarto-target-style");
      window.document.head.appendChild(linkStyle);
    }
    while (linkStyle.firstChild) {
      linkStyle.removeChild(linkStyle.firstChild);
    }
    if (topOffset > 0) {
      linkStyle.appendChild(
        window.document.createTextNode(`
      section:target::before {
        content: "";
        display: block;
        height: ${topOffset}px;
        margin: -${topOffset}px 0 0;
      }`)
      );
    }
    if (init) {
      window.dispatchEvent(headroomChanged);
    }
    init = true;
  }

  // initialize headroom
  var header = window.document.querySelector("#quarto-header");
  if (header && window.Headroom) {
    const headroom = new window.Headroom(header, {
      tolerance: 5,
      onPin: function () {
        const sidebars = window.document.querySelectorAll(
          ".sidebar, .headroom-target"
        );
        sidebars.forEach((sidebar) => {
          sidebar.classList.remove("sidebar-unpinned");
        });
        updateDocumentOffset();
      },
      onUnpin: function () {
        const sidebars = window.document.querySelectorAll(
          ".sidebar, .headroom-target"
        );
        sidebars.forEach((sidebar) => {
          sidebar.classList.add("sidebar-unpinned");
        });
        updateDocumentOffset();
      },
    });
    headroom.init();

    let frozen = false;
    window.quartoToggleHeadroom = function () {
      if (frozen) {
        headroom.unfreeze();
        frozen = false;
      } else {
        headroom.freeze();
        frozen = true;
      }
    };
  }

  window.addEventListener(
    "hashchange",
    function (e) {
      if (
        getComputedStyle(document.documentElement).scrollBehavior !== "smooth"
      ) {
        window.scrollTo(0, window.pageYOffset - headerOffset());
      }
    },
    false
  );

  // Observe size changed for the header
  const headerEl = window.document.querySelector("header.fixed-top");
  if (headerEl && window.ResizeObserver) {
    const observer = new window.ResizeObserver(
      updateDocumentOffsetWithoutAnimation
    );
    observer.observe(headerEl, {
      attributes: true,
      childList: true,
      characterData: true,
    });
  } else {
    window.addEventListener(
      "resize",
      throttle(updateDocumentOffsetWithoutAnimation, 50)
    );
  }
  setTimeout(updateDocumentOffsetWithoutAnimation, 250);

  // fixup index.html links if we aren't on the filesystem
  if (window.location.protocol !== "file:") {
    const links = window.document.querySelectorAll("a");
    for (let i = 0; i < links.length; i++) {
      if (links[i].href) {
        links[i].href = links[i].href.replace(/\/index\.html/, "/");
      }
    }

    // Fixup any sharing links that require urls
    // Append url to any sharing urls
    const sharingLinks = window.document.querySelectorAll(
      "a.sidebar-tools-main-item"
    );
    for (let i = 0; i < sharingLinks.length; i++) {
      const sharingLink = sharingLinks[i];
      const href = sharingLink.getAttribute("href");
      if (href) {
        sharingLink.setAttribute(
          "href",
          href.replace("|url|", window.location.href)
        );
      }
    }

    // Scroll the active navigation item into view, if necessary
    const navSidebar = window.document.querySelector("nav#quarto-sidebar");
    if (navSidebar) {
      // Find the active item
      const activeItem = navSidebar.querySelector("li.sidebar-item a.active");
      if (activeItem) {
        // Wait for the scroll height and height to resolve by observing size changes on the
        // nav element that is scrollable
        const resizeObserver = new ResizeObserver((_entries) => {
          // The bottom of the element
          const elBottom = activeItem.offsetTop;
          const viewBottom = navSidebar.scrollTop + navSidebar.clientHeight;

          // The element height and scroll height are the same, then we are still loading
          if (viewBottom !== navSidebar.scrollHeight) {
            // Determine if the item isn't visible and scroll to it
            if (elBottom >= viewBottom) {
              navSidebar.scrollTop = elBottom;
            }

            // stop observing now since we've completed the scroll
            resizeObserver.unobserve(navSidebar);
          }
        });
        resizeObserver.observe(navSidebar);
      }
    }
  }
});
