const kQueryArg = "q";
const kResultsArg = "show-results";

// If items don't provide a URL, then both the navigator and the onSelect
// function aren't called (and therefore, the default implementation is used)
//
// We're using this sentinel URL to signal to those handlers that this
// item is a more item (along with the type) and can be handled appropriately
const kItemTypeMoreHref = "0767FDFD-0422-4E5A-BC8A-3BE11E5BBA05";

window.document.addEventListener("DOMContentLoaded", function (_event) {
  // Ensure that search is available on this page. If it isn't,
  // should return early and not do anything
  var searchEl = window.document.getElementById("quarto-search");
  if (!searchEl) return;

  const { autocomplete } = window["@algolia/autocomplete-js"];

  let quartoSearchOptions = {};
  let language = {};
  const searchOptionEl = window.document.getElementById(
    "quarto-search-options"
  );
  if (searchOptionEl) {
    const jsonStr = searchOptionEl.textContent;
    quartoSearchOptions = JSON.parse(jsonStr);
    language = quartoSearchOptions.language;
  }

  // note the search mode
  if (quartoSearchOptions.type === "overlay") {
    searchEl.classList.add("type-overlay");
  } else {
    searchEl.classList.add("type-textbox");
  }

  // Used to determine highlighting behavior for this page
  // A `q` query param is expected when the user follows a search
  // to this page
  const currentUrl = new URL(window.location);
  const query = currentUrl.searchParams.get(kQueryArg);
  const showSearchResults = currentUrl.searchParams.get(kResultsArg);
  const mainEl = window.document.querySelector("main");

  // highlight matches on the page
  if (query !== null && mainEl) {
    // perform any highlighting
    highlight(escapeRegExp(query), mainEl);

    // fix up the URL to remove the q query param
    const replacementUrl = new URL(window.location);
    replacementUrl.searchParams.delete(kQueryArg);
    window.history.replaceState({}, "", replacementUrl);
  }

  // function to clear highlighting on the page when the search query changes
  // (e.g. if the user edits the query or clears it)
  let highlighting = true;
  const resetHighlighting = (searchTerm) => {
    if (mainEl && highlighting && query !== null && searchTerm !== query) {
      clearHighlight(query, mainEl);
      highlighting = false;
    }
  };

  // Clear search highlighting when the user scrolls sufficiently
  const resetFn = () => {
    resetHighlighting("");
    window.removeEventListener("quarto-hrChanged", resetFn);
    window.removeEventListener("quarto-sectionChanged", resetFn);
  };

  // Register this event after the initial scrolling and settling of events
  // on the page
  window.addEventListener("quarto-hrChanged", resetFn);
  window.addEventListener("quarto-sectionChanged", resetFn);

  // Responsively switch to overlay mode if the search is present on the navbar
  // Note that switching the sidebar to overlay mode requires more coordinate (not just
  // the media query since we generate different HTML for sidebar overlays than we do
  // for sidebar input UI)
  const detachedMediaQuery =
    quartoSearchOptions.type === "overlay" ? "all" : "(max-width: 991px)";

  // If configured, include the analytics client to send insights
  const plugins = configurePlugins(quartoSearchOptions);

  let lastState = null;
  const { setIsOpen, setQuery, setCollections } = autocomplete({
    container: searchEl,
    detachedMediaQuery: detachedMediaQuery,
    defaultActiveItemId: 0,
    panelContainer: "#quarto-search-results",
    panelPlacement: quartoSearchOptions["panel-placement"],
    debug: false,
    openOnFocus: true,
    plugins,
    classNames: {
      form: "d-flex",
    },
    translations: {
      clearButtonTitle: language["search-clear-button-title"],
      detachedCancelButtonText: language["search-detached-cancel-button-title"],
      submitButtonTitle: language["search-submit-button-title"],
    },
    initialState: {
      query,
    },
    getItemUrl({ item }) {
      return item.href;
    },
    onStateChange({ state }) {
      // Perhaps reset highlighting
      resetHighlighting(state.query);

      // If the panel just opened, ensure the panel is positioned properly
      if (state.isOpen) {
        if (lastState && !lastState.isOpen) {
          setTimeout(() => {
            positionPanel(quartoSearchOptions["panel-placement"]);
          }, 150);
        }
      }

      // Perhaps show the copy link
      showCopyLink(state.query, quartoSearchOptions);

      lastState = state;
    },
    reshape({ sources, state }) {
      return sources.map((source) => {
        try {
          const items = source.getItems();

          // Validate the items
          validateItems(items);

          // group the items by document
          const groupedItems = new Map();
          items.forEach((item) => {
            const hrefParts = item.href.split("#");
            const baseHref = hrefParts[0];
            const isDocumentItem = hrefParts.length === 1;

            const items = groupedItems.get(baseHref);
            if (!items) {
              groupedItems.set(baseHref, [item]);
            } else {
              // If the href for this item matches the document
              // exactly, place this item first as it is the item that represents
              // the document itself
              if (isDocumentItem) {
                items.unshift(item);
              } else {
                items.push(item);
              }
              groupedItems.set(baseHref, items);
            }
          });

          const reshapedItems = [];
          let count = 1;
          for (const [_key, value] of groupedItems) {
            const firstItem = value[0];
            reshapedItems.push({
              ...firstItem,
              type: kItemTypeDoc,
            });

            const collapseMatches = quartoSearchOptions["collapse-after"];
            const collapseCount =
              typeof collapseMatches === "number" ? collapseMatches : 1;

            if (value.length > 1) {
              const target = `search-more-${count}`;
              const isExpanded =
                state.context.expanded &&
                state.context.expanded.includes(target);

              const remainingCount = value.length - collapseCount;

              for (let i = 1; i < value.length; i++) {
                if (collapseMatches && i === collapseCount) {
                  reshapedItems.push({
                    target,
                    title: isExpanded
                      ? language["search-hide-matches-text"]
                      : remainingCount === 1
                      ? `${remainingCount} ${language["search-more-match-text"]}`
                      : `${remainingCount} ${language["search-more-matches-text"]}`,
                    type: kItemTypeMore,
                    href: kItemTypeMoreHref,
                  });
                }

                if (isExpanded || !collapseMatches || i < collapseCount) {
                  reshapedItems.push({
                    ...value[i],
                    type: kItemTypeItem,
                    target,
                  });
                }
              }
            }
            count += 1;
          }

          return {
            ...source,
            getItems() {
              return reshapedItems;
            },
          };
        } catch (error) {
          // Some form of error occurred
          return {
            ...source,
            getItems() {
              return [
                {
                  title: error.name || "An Error Occurred While Searching",
                  text:
                    error.message ||
                    "An unknown error occurred while attempting to perform the requested search.",
                  type: kItemTypeError,
                },
              ];
            },
          };
        }
      });
    },
    navigator: {
      navigate({ itemUrl }) {
        if (itemUrl !== offsetURL(kItemTypeMoreHref)) {
          window.location.assign(itemUrl);
        }
      },
      navigateNewTab({ itemUrl }) {
        if (itemUrl !== offsetURL(kItemTypeMoreHref)) {
          const windowReference = window.open(itemUrl, "_blank", "noopener");
          if (windowReference) {
            windowReference.focus();
          }
        }
      },
      navigateNewWindow({ itemUrl }) {
        if (itemUrl !== offsetURL(kItemTypeMoreHref)) {
          window.open(itemUrl, "_blank", "noopener");
        }
      },
    },
    getSources({ state, setContext, setActiveItemId, refresh }) {
      return [
        {
          sourceId: "documents",
          getItemUrl({ item }) {
            if (item.href) {
              return offsetURL(item.href);
            } else {
              return undefined;
            }
          },
          onSelect({
            item,
            state,
            setContext,
            setIsOpen,
            setActiveItemId,
            refresh,
          }) {
            if (item.type === kItemTypeMore) {
              toggleExpanded(item, state, setContext, setActiveItemId, refresh);

              // Toggle more
              setIsOpen(true);
            }
          },
          getItems({ query }) {
            if (query === null || query === "") {
              return [];
            }

            const limit = quartoSearchOptions.limit;
            if (quartoSearchOptions.algolia) {
              return algoliaSearch(query, limit, quartoSearchOptions.algolia);
            } else {
              // Fuse search options
              const fuseSearchOptions = {
                isCaseSensitive: false,
                shouldSort: true,
                minMatchCharLength: 2,
                limit: limit,
              };

              return readSearchData().then(function (fuse) {
                return fuseSearch(query, fuse, fuseSearchOptions);
              });
            }
          },
          templates: {
            noResults({ createElement }) {
              const hasQuery = lastState.query;

              return createElement(
                "div",
                {
                  class: `quarto-search-no-results${
                    hasQuery ? "" : " no-query"
                  }`,
                },
                language["search-no-results-text"]
              );
            },
            header({ items, createElement }) {
              // count the documents
              const count = items.filter((item) => {
                return item.type === kItemTypeDoc;
              }).length;

              if (count > 0) {
                return createElement(
                  "div",
                  { class: "search-result-header" },
                  `${count} ${language["search-matching-documents-text"]}`
                );
              } else {
                return createElement(
                  "div",
                  { class: "search-result-header-no-results" },
                  ``
                );
              }
            },
            footer({ _items, createElement }) {
              if (
                quartoSearchOptions.algolia &&
                quartoSearchOptions.algolia["show-logo"]
              ) {
                const libDir = quartoSearchOptions.algolia["libDir"];
                const logo = createElement("img", {
                  src: offsetURL(
                    `${libDir}/quarto-search/search-by-algolia.svg`
                  ),
                  class: "algolia-search-logo",
                });
                return createElement(
                  "a",
                  { href: "http://www.algolia.com/" },
                  logo
                );
              }
            },

            item({ item, createElement }) {
              return renderItem(
                item,
                createElement,
                state,
                setActiveItemId,
                setContext,
                refresh
              );
            },
          },
        },
      ];
    },
  });

  window.quartoOpenSearch = () => {
    setIsOpen(false);
    setIsOpen(true);
    focusSearchInput();
  };

  // Remove the labeleledby attribute since it is pointing
  // to a non-existent label
  if (quartoSearchOptions.type === "overlay") {
    const inputEl = window.document.querySelector(
      "#quarto-search .aa-Autocomplete"
    );
    if (inputEl) {
      inputEl.removeAttribute("aria-labelledby");
    }
  }

  // If the main document scrolls dismiss the search results
  // (otherwise, since they're floating in the document they can scroll with the document)
  window.document.body.onscroll = () => {
    setIsOpen(false);
  };

  if (showSearchResults) {
    setIsOpen(true);
    focusSearchInput();
  }
});

function configurePlugins(quartoSearchOptions) {
  const autocompletePlugins = [];
  const algoliaOptions = quartoSearchOptions.algolia;
  if (
    algoliaOptions &&
    algoliaOptions["analytics-events"] &&
    algoliaOptions["search-only-api-key"] &&
    algoliaOptions["application-id"]
  ) {
    const apiKey = algoliaOptions["search-only-api-key"];
    const appId = algoliaOptions["application-id"];

    // Aloglia insights may not be loaded because they require cookie consent
    // Use deferred loading so events will start being recorded when/if consent
    // is granted.
    const algoliaInsightsDeferredPlugin = deferredLoadPlugin(() => {
      if (
        window.aa &&
        window["@algolia/autocomplete-plugin-algolia-insights"]
      ) {
        window.aa("init", {
          appId,
          apiKey,
          useCookie: true,
        });

        const { createAlgoliaInsightsPlugin } =
          window["@algolia/autocomplete-plugin-algolia-insights"];
        // Register the insights client
        const algoliaInsightsPlugin = createAlgoliaInsightsPlugin({
          insightsClient: window.aa,
          onItemsChange({ insights, insightsEvents }) {
            const events = insightsEvents.map((event) => {
              const maxEvents = event.objectIDs.slice(0, 20);
              return {
                ...event,
                objectIDs: maxEvents,
              };
            });

            insights.viewedObjectIDs(...events);
          },
        });
        return algoliaInsightsPlugin;
      }
    });

    // Add the plugin
    autocompletePlugins.push(algoliaInsightsDeferredPlugin);
    return autocompletePlugins;
  }
}

// For plugins that may not load immediately, create a wrapper
// plugin and forward events and plugin data once the plugin
// is initialized. This is useful for cases like cookie consent
// which may prevent the analytics insights event plugin from initializing
// immediately.
function deferredLoadPlugin(createPlugin) {
  let plugin = undefined;
  let subscribeObj = undefined;
  const wrappedPlugin = () => {
    if (!plugin && subscribeObj) {
      plugin = createPlugin();
      if (plugin && plugin.subscribe) {
        plugin.subscribe(subscribeObj);
      }
    }
    return plugin;
  };

  return {
    subscribe: (obj) => {
      subscribeObj = obj;
    },
    onStateChange: (obj) => {
      const plugin = wrappedPlugin();
      if (plugin && plugin.onStateChange) {
        plugin.onStateChange(obj);
      }
    },
    onSubmit: (obj) => {
      const plugin = wrappedPlugin();
      if (plugin && plugin.onSubmit) {
        plugin.onSubmit(obj);
      }
    },
    onReset: (obj) => {
      const plugin = wrappedPlugin();
      if (plugin && plugin.onReset) {
        plugin.onReset(obj);
      }
    },
    getSources: (obj) => {
      const plugin = wrappedPlugin();
      if (plugin && plugin.getSources) {
        return plugin.getSources(obj);
      } else {
        return Promise.resolve([]);
      }
    },
    data: (obj) => {
      const plugin = wrappedPlugin();
      if (plugin && plugin.data) {
        plugin.data(obj);
      }
    },
  };
}

function validateItems(items) {
  // Validate the first item
  if (items.length > 0) {
    const item = items[0];
    const missingFields = [];
    if (item.href == undefined) {
      missingFields.push("href");
    }
    if (!item.title == undefined) {
      missingFields.push("title");
    }
    if (!item.text == undefined) {
      missingFields.push("text");
    }

    if (missingFields.length === 1) {
      throw {
        name: `Error: Search index is missing the <code>${missingFields[0]}</code> field.`,
        message: `The items being returned for this search do not include all the required fields. Please ensure that your index items include the <code>${missingFields[0]}</code> field or use <code>index-fields</code> in your <code>_quarto.yml</code> file to specify the field names.`,
      };
    } else if (missingFields.length > 1) {
      const missingFieldList = missingFields
        .map((field) => {
          return `<code>${field}</code>`;
        })
        .join(", ");

      throw {
        name: `Error: Search index is missing the following fields: ${missingFieldList}.`,
        message: `The items being returned for this search do not include all the required fields. Please ensure that your index items includes the following fields: ${missingFieldList}, or use <code>index-fields</code> in your <code>_quarto.yml</code> file to specify the field names.`,
      };
    }
  }
}

let lastQuery = null;
function showCopyLink(query, options) {
  const language = options.language;
  lastQuery = query;
  // Insert share icon
  const inputSuffixEl = window.document.body.querySelector(
    ".aa-Form .aa-InputWrapperSuffix"
  );

  if (inputSuffixEl) {
    let copyButtonEl = window.document.body.querySelector(
      ".aa-Form .aa-InputWrapperSuffix .aa-CopyButton"
    );

    if (copyButtonEl === null) {
      copyButtonEl = window.document.createElement("button");
      copyButtonEl.setAttribute("class", "aa-CopyButton");
      copyButtonEl.setAttribute("type", "button");
      copyButtonEl.setAttribute("title", language["search-copy-link-title"]);
      copyButtonEl.onmousedown = (e) => {
        e.preventDefault();
        e.stopPropagation();
      };

      const linkIcon = "bi-clipboard";
      const checkIcon = "bi-check2";

      const shareIconEl = window.document.createElement("i");
      shareIconEl.setAttribute("class", `bi ${linkIcon}`);
      copyButtonEl.appendChild(shareIconEl);
      inputSuffixEl.prepend(copyButtonEl);

      const clipboard = new window.ClipboardJS(".aa-CopyButton", {
        text: function (_trigger) {
          const copyUrl = new URL(window.location);
          copyUrl.searchParams.set(kQueryArg, lastQuery);
          copyUrl.searchParams.set(kResultsArg, "1");
          return copyUrl.toString();
        },
      });
      clipboard.on("success", function (e) {
        // Focus the input

        // button target
        const button = e.trigger;
        const icon = button.querySelector("i.bi");

        // flash "checked"
        icon.classList.add(checkIcon);
        icon.classList.remove(linkIcon);
        setTimeout(function () {
          icon.classList.remove(checkIcon);
          icon.classList.add(linkIcon);
        }, 1000);
      });
    }

    // If there is a query, show the link icon
    if (copyButtonEl) {
      if (lastQuery && options["copy-button"]) {
        copyButtonEl.style.display = "flex";
      } else {
        copyButtonEl.style.display = "none";
      }
    }
  }
}

/* Search Index Handling */
// create the index
var fuseIndex = undefined;
async function readSearchData() {
  // Initialize the search index on demand
  if (fuseIndex === undefined) {
    // create fuse index
    const options = {
      keys: [
        { name: "title", weight: 20 },
        { name: "section", weight: 20 },
        { name: "text", weight: 10 },
      ],
      ignoreLocation: true,
      threshold: 0.1,
    };
    const fuse = new window.Fuse([], options);

    // fetch the main search.json
    const response = await fetch(offsetURL("search.json"));
    if (response.status == 200) {
      return response.json().then(function (searchDocs) {
        searchDocs.forEach(function (searchDoc) {
          fuse.add(searchDoc);
        });
        fuseIndex = fuse;
        return fuseIndex;
      });
    } else {
      return Promise.reject(
        new Error(
          "Unexpected status from search index request: " + response.status
        )
      );
    }
  }
  return fuseIndex;
}

function inputElement() {
  return window.document.body.querySelector(".aa-Form .aa-Input");
}

function focusSearchInput() {
  setTimeout(() => {
    const inputEl = inputElement();
    if (inputEl) {
      inputEl.focus();
    }
  }, 50);
}

/* Panels */
const kItemTypeDoc = "document";
const kItemTypeMore = "document-more";
const kItemTypeItem = "document-item";
const kItemTypeError = "error";

function renderItem(
  item,
  createElement,
  state,
  setActiveItemId,
  setContext,
  refresh
) {
  switch (item.type) {
    case kItemTypeDoc:
      return createDocumentCard(
        createElement,
        "file-richtext",
        item.title,
        item.section,
        item.text,
        item.href
      );
    case kItemTypeMore:
      return createMoreCard(
        createElement,
        item,
        state,
        setActiveItemId,
        setContext,
        refresh
      );
    case kItemTypeItem:
      return createSectionCard(
        createElement,
        item.section,
        item.text,
        item.href
      );
    case kItemTypeError:
      return createErrorCard(createElement, item.title, item.text);
    default:
      return undefined;
  }
}

function createDocumentCard(createElement, icon, title, section, text, href) {
  const iconEl = createElement("i", {
    class: `bi bi-${icon} search-result-icon`,
  });
  const titleEl = createElement("p", { class: "search-result-title" }, title);
  const titleContainerEl = createElement(
    "div",
    { class: "search-result-title-container" },
    [iconEl, titleEl]
  );

  const textEls = [];
  if (section) {
    const sectionEl = createElement(
      "p",
      { class: "search-result-section" },
      section
    );
    textEls.push(sectionEl);
  }
  const descEl = createElement("p", {
    class: "search-result-text",
    dangerouslySetInnerHTML: {
      __html: text,
    },
  });
  textEls.push(descEl);

  const textContainerEl = createElement(
    "div",
    { class: "search-result-text-container" },
    textEls
  );

  const containerEl = createElement(
    "div",
    {
      class: "search-result-container",
    },
    [titleContainerEl, textContainerEl]
  );

  const linkEl = createElement(
    "a",
    {
      href: offsetURL(href),
      class: "search-result-link",
    },
    containerEl
  );

  const classes = ["search-result-doc", "search-item"];
  if (!section) {
    classes.push("document-selectable");
  }

  return createElement(
    "div",
    {
      class: classes.join(" "),
    },
    linkEl
  );
}

function createMoreCard(
  createElement,
  item,
  state,
  setActiveItemId,
  setContext,
  refresh
) {
  const moreCardEl = createElement(
    "div",
    {
      class: "search-result-more search-item",
      onClick: (e) => {
        // Handle expanding the sections by adding the expanded
        // section to the list of expanded sections
        toggleExpanded(item, state, setContext, setActiveItemId, refresh);
        e.stopPropagation();
      },
    },
    item.title
  );

  return moreCardEl;
}

function toggleExpanded(item, state, setContext, setActiveItemId, refresh) {
  const expanded = state.context.expanded || [];
  if (expanded.includes(item.target)) {
    setContext({
      expanded: expanded.filter((target) => target !== item.target),
    });
  } else {
    setContext({ expanded: [...expanded, item.target] });
  }

  refresh();
  setActiveItemId(item.__autocomplete_id);
}

function createSectionCard(createElement, section, text, href) {
  const sectionEl = createSection(createElement, section, text, href);
  return createElement(
    "div",
    {
      class: "search-result-doc-section search-item",
    },
    sectionEl
  );
}

function createSection(createElement, title, text, href) {
  const descEl = createElement("p", {
    class: "search-result-text",
    dangerouslySetInnerHTML: {
      __html: text,
    },
  });

  const titleEl = createElement("p", { class: "search-result-section" }, title);
  const linkEl = createElement(
    "a",
    {
      href: offsetURL(href),
      class: "search-result-link",
    },
    [titleEl, descEl]
  );
  return linkEl;
}

function createErrorCard(createElement, title, text) {
  const descEl = createElement("p", {
    class: "search-error-text",
    dangerouslySetInnerHTML: {
      __html: text,
    },
  });

  const titleEl = createElement("p", {
    class: "search-error-title",
    dangerouslySetInnerHTML: {
      __html: `<i class="bi bi-exclamation-circle search-error-icon"></i> ${title}`,
    },
  });
  const errorEl = createElement("div", { class: "search-error" }, [
    titleEl,
    descEl,
  ]);
  return errorEl;
}

function positionPanel(pos) {
  const panelEl = window.document.querySelector(
    "#quarto-search-results .aa-Panel"
  );
  const inputEl = window.document.querySelector(
    "#quarto-search .aa-Autocomplete"
  );

  if (panelEl && inputEl) {
    panelEl.style.top = `${Math.round(panelEl.offsetTop)}px`;
    if (pos === "start") {
      panelEl.style.left = `${Math.round(inputEl.left)}px`;
    } else {
      panelEl.style.right = `${Math.round(inputEl.offsetRight)}px`;
    }
  }
}

/* Highlighting */
// highlighting functions
function highlightMatch(query, text) {
  if (text) {
    const start = text.toLowerCase().indexOf(query.toLowerCase());
    if (start !== -1) {
      const startMark = "<mark class='search-match'>";
      const endMark = "</mark>";

      const end = start + query.length;
      text =
        text.slice(0, start) +
        startMark +
        text.slice(start, end) +
        endMark +
        text.slice(end);
      const startInfo = clipStart(text, start);
      const endInfo = clipEnd(
        text,
        startInfo.position + startMark.length + endMark.length
      );
      text =
        startInfo.prefix +
        text.slice(startInfo.position, endInfo.position) +
        endInfo.suffix;

      return text;
    } else {
      return text;
    }
  } else {
    return text;
  }
}

function clipStart(text, pos) {
  const clipStart = pos - 50;
  if (clipStart < 0) {
    // This will just return the start of the string
    return {
      position: 0,
      prefix: "",
    };
  } else {
    // We're clipping before the start of the string, walk backwards to the first space.
    const spacePos = findSpace(text, pos, -1);
    return {
      position: spacePos.position,
      prefix: "",
    };
  }
}

function clipEnd(text, pos) {
  const clipEnd = pos + 200;
  if (clipEnd > text.length) {
    return {
      position: text.length,
      suffix: "",
    };
  } else {
    const spacePos = findSpace(text, clipEnd, 1);
    return {
      position: spacePos.position,
      suffix: spacePos.clipped ? "…" : "",
    };
  }
}

function findSpace(text, start, step) {
  let stepPos = start;
  while (stepPos > -1 && stepPos < text.length) {
    const char = text[stepPos];
    if (char === " " || char === "," || char === ":") {
      return {
        position: step === 1 ? stepPos : stepPos - step,
        clipped: stepPos > 1 && stepPos < text.length,
      };
    }
    stepPos = stepPos + step;
  }

  return {
    position: stepPos - step,
    clipped: false,
  };
}

// removes highlighting as implemented by the mark tag
function clearHighlight(searchterm, el) {
  const childNodes = el.childNodes;
  for (let i = childNodes.length - 1; i >= 0; i--) {
    const node = childNodes[i];
    if (node.nodeType === Node.ELEMENT_NODE) {
      if (
        node.tagName === "MARK" &&
        node.innerText.toLowerCase() === searchterm.toLowerCase()
      ) {
        el.replaceChild(document.createTextNode(node.innerText), node);
      } else {
        clearHighlight(searchterm, node);
      }
    }
  }
}

function escapeRegExp(string) {
  return string.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"); // $& means the whole matched string
}

// highlight matches
function highlight(term, el) {
  const termRegex = new RegExp(term, "ig");
  const childNodes = el.childNodes;

  // walk back to front avoid mutating elements in front of us
  for (let i = childNodes.length - 1; i >= 0; i--) {
    const node = childNodes[i];

    if (node.nodeType === Node.TEXT_NODE) {
      // Search text nodes for text to highlight
      const text = node.nodeValue;

      let startIndex = 0;
      let matchIndex = text.search(termRegex);
      if (matchIndex > -1) {
        const markFragment = document.createDocumentFragment();
        while (matchIndex > -1) {
          const prefix = text.slice(startIndex, matchIndex);
          markFragment.appendChild(document.createTextNode(prefix));

          const mark = document.createElement("mark");
          mark.appendChild(
            document.createTextNode(
              text.slice(matchIndex, matchIndex + term.length)
            )
          );
          markFragment.appendChild(mark);

          startIndex = matchIndex + term.length;
          matchIndex = text.slice(startIndex).search(new RegExp(term, "ig"));
          if (matchIndex > -1) {
            matchIndex = startIndex + matchIndex;
          }
        }
        if (startIndex < text.length) {
          markFragment.appendChild(
            document.createTextNode(text.slice(startIndex, text.length))
          );
        }

        el.replaceChild(markFragment, node);
      }
    } else if (node.nodeType === Node.ELEMENT_NODE) {
      // recurse through elements
      highlight(term, node);
    }
  }
}

/* Link Handling */
// get the offset from this page for a given site root relative url
function offsetURL(url) {
  var offset = getMeta("quarto:offset");
  return offset ? offset + url : url;
}

// read a meta tag value
function getMeta(metaName) {
  var metas = window.document.getElementsByTagName("meta");
  for (let i = 0; i < metas.length; i++) {
    if (metas[i].getAttribute("name") === metaName) {
      return metas[i].getAttribute("content");
    }
  }
  return "";
}

function algoliaSearch(query, limit, algoliaOptions) {
  const { getAlgoliaResults } = window["@algolia/autocomplete-preset-algolia"];

  const applicationId = algoliaOptions["application-id"];
  const searchOnlyApiKey = algoliaOptions["search-only-api-key"];
  const indexName = algoliaOptions["index-name"];
  const indexFields = algoliaOptions["index-fields"];
  const searchClient = window.algoliasearch(applicationId, searchOnlyApiKey);
  const searchParams = algoliaOptions["params"];
  const searchAnalytics = !!algoliaOptions["analytics-events"];

  return getAlgoliaResults({
    searchClient,
    queries: [
      {
        indexName: indexName,
        query,
        params: {
          hitsPerPage: limit,
          clickAnalytics: searchAnalytics,
          ...searchParams,
        },
      },
    ],
    transformResponse: (response) => {
      if (!indexFields) {
        return response.hits.map((hit) => {
          return hit.map((item) => {
            return {
              ...item,
              text: highlightMatch(query, item.text),
            };
          });
        });
      } else {
        const remappedHits = response.hits.map((hit) => {
          return hit.map((item) => {
            const newItem = { ...item };
            ["href", "section", "title", "text"].forEach((keyName) => {
              const mappedName = indexFields[keyName];
              if (
                mappedName &&
                item[mappedName] !== undefined &&
                mappedName !== keyName
              ) {
                newItem[keyName] = item[mappedName];
                delete newItem[mappedName];
              }
            });
            newItem.text = highlightMatch(query, newItem.text);
            return newItem;
          });
        });
        return remappedHits;
      }
    },
  });
}

function fuseSearch(query, fuse, fuseOptions) {
  return fuse.search(query, fuseOptions).map((result) => {
    const addParam = (url, name, value) => {
      const anchorParts = url.split("#");
      const baseUrl = anchorParts[0];
      const sep = baseUrl.search("\\?") > 0 ? "&" : "?";
      anchorParts[0] = baseUrl + sep + name + "=" + value;
      return anchorParts.join("#");
    };

    return {
      title: result.item.title,
      section: result.item.section,
      href: addParam(result.item.href, kQueryArg, query),
      text: highlightMatch(query, result.item.text),
    };
  });
}
