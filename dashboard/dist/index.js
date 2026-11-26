/* MediaRAG dashboard tab: corpus search with thumbnails + status chip.
 * No build step — plain IIFE, React from the host SDK. */
(function () {
  "use strict";
  var SDK = window.__HERMES_PLUGIN_SDK__;
  var React = SDK.React;
  var h = React.createElement;
  var Card = SDK.components.Card,
      CardHeader = SDK.components.CardHeader,
      CardTitle = SDK.components.CardTitle,
      CardContent = SDK.components.CardContent,
      Badge = SDK.components.Badge,
      Button = SDK.components.Button,
      Input = SDK.components.Input;

  var KIND_LABEL = { text: "text", image: "image", audio: "audio" };

  function fmtSpan(hit) {
    if (hit.span_end && hit.span_start >= 0 && hit.span_end > hit.span_start)
      return "  " + hit.span_start.toFixed(1) + "–" + hit.span_end.toFixed(1) + "s";
    return "";
  }

  function Hit(props) {
    var hit = props.hit;
    var thumb = hit.thumbnail
      ? h("img", {
          src: hit.thumbnail,
          alt: hit.uri.split("/").pop(),
          className: "h-24 w-24 rounded-md border object-cover",
        })
      : null;
    return h(Card, { key: hit.uri + ":" + hit.span_start + ":" + hit.span_end },
      h(CardContent, { className: "flex items-start gap-3 py-3" },
        thumb,
        h("div", { className: "min-w-0 flex-1" },
          h("div", { className: "mb-1 flex flex-wrap items-center gap-2" },
            h(Badge, { variant: "secondary" }, KIND_LABEL[hit.kind] || hit.kind),
            h("span", { className: "text-xs text-muted-foreground" },
              "score " + hit.score.toFixed(3) + fmtSpan(hit)),
          ),
          h("div", { className: "truncate text-xs text-muted-foreground", title: hit.uri }, hit.uri),
          h("div", { className: "mt-1 text-sm" }, hit.snippet),
        ),
      ),
    );
  }

  function App() {
    var state = SDK.hooks.useState({ q: "", kind: "", hits: null, busy: false, err: null, status: null });
    var st = state[0], set = state[1];

    SDK.hooks.useEffect(function () {
      SDK.fetchJSON("/api/plugins/mediarag/status")
        .then(function (s) { set(function (p) { return Object.assign({}, p, { status: s }); }); })
        .catch(function (e) { set(function (p) { return Object.assign({}, p, { err: String(e) }); }); });
    }, []);

    function run(ev) {
      ev && ev.preventDefault();
      var q = st.q.trim();
      if (!q) return;
      set(function (p) { return Object.assign({}, p, { busy: true, err: null }); });
      var url = "/api/plugins/mediarag/search?q=" + encodeURIComponent(q) +
        (st.kind ? "&kind=" + st.kind : "") + "&k=8";
      SDK.fetchJSON(url)
        .then(function (r) {
          set(function (p) { return Object.assign({}, p, { busy: false, hits: r.results || [] }); });
        })
        .catch(function (e) {
          set(function (p) { return Object.assign({}, p, { busy: false, err: String(e), hits: [] }); });
        });
      return false;
    }

    var s = st.status;
    var chip = s
      ? h("span", { className: "text-xs text-muted-foreground" },
          (s.total || 0).toLocaleString() + " chunks · " + (s.uris || 0).toLocaleString() + " sources")
      : h("span", { className: "text-xs text-muted-foreground" }, "…");

    var results = st.hits === null ? null : (st.hits.length
      ? h("div", { className: "space-y-3" }, st.hits.map(Hit))
      : h("p", { className: "text-sm text-muted-foreground" }, "No hits."));

    var kindSel = h("select", {
      value: st.kind,
      onChange: function (e) { set(function (p) { return Object.assign({}, p, { kind: e.target.value }); }); },
      className: "w-32 rounded-md border bg-transparent px-2 text-sm outline-none",
    },
      h("option", { value: "" }, "all kinds"),
      h("option", { value: "text" }, "text"),
      h("option", { value: "image" }, "image"),
      h("option", { value: "audio" }, "audio"),
      h("option", { value: "video" }, "video frames"),
    );

    return h("div", { className: "mx-auto max-w-3xl space-y-5 p-6" },
      h(Card, null,
        h(CardHeader, null,
          h(CardTitle, { className: "flex items-center justify-between" },
            "MediaRAG", chip)),
      ),
      h("form", { onSubmit: run, className: "flex gap-2" },
        h(Input, {
          value: st.q,
          onChange: function (e) { set(function (p) { return Object.assign({}, p, { q: e.target.value }); }); },
          placeholder: "Search the corpus by meaning…",
          className: "flex-1",
          autoFocus: true,
        }),
        kindSel,
        h(Button, { type: "submit", disabled: st.busy }, st.busy ? "…" : "Search"),
      ),
      st.err && h("p", { className: "text-sm text-destructive" }, "error: " + st.err),
      results,
    );
  }

  window.__HERMES_PLUGINS__.register("mediarag", App);
})();
