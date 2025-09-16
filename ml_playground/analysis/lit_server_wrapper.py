from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Protocol, cast


class WSGIApp(Protocol):
    def __call__(
        self,
        _environ: Dict[str, Any],
        _start_response: Callable[[str, List[tuple[str, str]]], None],
    ) -> Iterable[bytes]: ...


def _build_demo_components() -> tuple[Dict[str, object], Dict[str, object]]:
    """Create minimal dataset and echo model, mirroring lit_integration."""
    # Lazy imports
    from lit_nlp.api import dataset as lit_dataset  # type: ignore
    from lit_nlp.api import model as lit_model  # type: ignore
    from lit_nlp.api import types as lit_types  # type: ignore

    # Small sample data (optionally read bundestag_char input.txt if available)
    samples: List[str] = [
        "Nächste Rednerin ist die Vorsitzende der AfD-Fraktion, Dr. Alice Weidel.",
        "Herr Präsident, liebe Kolleginnen und Kollegen, wir beraten heute wichtige Vorlagen.",
        "(Beifall bei der SPD)",
        "Die Bundesregierung handelt entschlossen.",
        "Applaus bei der CDU/CSU.",
        "Vielen Dank. — Zur Geschäftsordnung hat der Abgeordnete das Wort.",
        "Wir müssen die Inflation bekämpfen und Familien entlasten.",
        "Das Wort hat nun die Bundeskanzlerin.",
        "Meine Damen und Herren, die Lage ist ernst, aber beherrschbar.",
        "(Heiterkeit) Der nächste Redner folgt.",
    ]
    try:
        base_dir = Path(__file__).resolve().parents[1]
        exp_dir = base_dir / "experiments" / "bundestag_char"
        input_path = exp_dir / "datasets" / "input.txt"
        if input_path.exists():
            text = input_path.read_text(encoding="utf-8", errors="ignore")
            file_lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
            if file_lines:
                samples = file_lines[:10]
    except Exception:
        pass

    class BundestagTextDataset(lit_dataset.Dataset):
        def __init__(self, sents: Iterable[str]):
            self._examples: List[Mapping[str, str]] = [{"text": s} for s in sents]

        def spec(self) -> Dict[str, object]:  # type: ignore[override]
            return {"text": lit_types.TextSegment()}

        def __len__(self) -> int:
            return len(self._examples)

        def __iter__(self):
            return iter(self._examples)

    class EchoModel(lit_model.Model):
        def input_spec(self) -> Dict[str, object]:  # type: ignore[override]
            return {"text": lit_types.TextSegment()}

        def output_spec(self) -> Dict[str, object]:  # type: ignore[override]
            return {"generated": lit_types.TextSegment()}

        def predict(
            self, _inputs: Iterable[Mapping[str, object]], **kwargs: object
        ) -> List[Mapping[str, object]]:
            outs: List[Mapping[str, object]] = []
            for ex in _inputs:
                s = str(ex.get("text", ""))
                gen = s + "\n\n[echo] " + s[::-1]
                outs.append({"generated": gen})
            return outs

    datasets: Dict[str, object] = {
        "bundestag_char_sample": BundestagTextDataset(samples)
    }
    models: Dict[str, object] = {"echo_model": EchoModel()}
    # Explicit casts ensure invariance requirements for Dict don't trip type checkers
    return cast(Dict[str, object], models), cast(Dict[str, object], datasets)


def main(host: str, port: int, open_browser: bool) -> None:
    # Import lit server implementation in a robust way
    import importlib

    server_module = None
    for mod in (
        "lit_nlp.server",
        "lit_nlp.dev_server",
        "lit_nlp.runtime.server",
        "lit_nlp.lib.server",
    ):
        try:
            server_module = importlib.import_module(mod)
            break
        except Exception:
            continue
    if server_module is None:
        try:
            import lit_nlp  # type: ignore

            lit_ver = getattr(lit_nlp, "__version__", "<unknown>")
        except Exception:
            lit_ver = "<unavailable>"
        raise RuntimeError(
            f"Could not import LIT server module; installed lit-nlp={lit_ver}"
        )

    models, datasets = _build_demo_components()

    # Try constructing the LIT app/server object
    # The class is commonly exposed as `Server` in server modules.
    if hasattr(server_module, "Server"):
        app = server_module.Server(models, datasets)  # type: ignore[attr-defined]
    else:
        # Fallback: some versions may expose a factory
        factory = getattr(server_module, "create_server", None)
        if callable(factory):
            app = factory(models, datasets)  # type: ignore[misc]
        else:
            raise RuntimeError(
                "No Server class or create_server() in LIT server module"
            )

    url = f"http://{host}:{port if port else '<auto>'}"
    print("[LIT] Registered models:", ", ".join(models.keys()))
    print("[LIT] Registered datasets:", ", ".join(datasets.keys()))
    print(f"[LIT] Starting server at {url}")
    sys.stdout.flush()

    # Try common start methods first
    for name in ("serve", "run", "start", "launch"):
        fn = getattr(server_module, name, None)
        if callable(fn):
            try:
                fn(app, port=port, host=host, open_browser=open_browser)
                return
            except TypeError:
                try:
                    fn(app, port, host)
                    return
                except Exception:
                    try:
                        fn(app, host, port)
                        return
                    except Exception:
                        pass

    # Try app-level starters
    for name in ("serve", "run", "start", "launch", "serve_forever"):
        fn = getattr(app, name, None)
        if callable(fn):
            # Some versions expose app.serve() with zero-arg signature; call it directly
            try:
                fn()
                return
            except TypeError:
                try:
                    fn(port=port, host=host, open_browser=open_browser)
                    return
                except Exception:
                    try:
                        fn(port, host)
                        return
                    except Exception:
                        try:
                            fn(host, port)
                            return
                        except Exception:
                            pass

    # Dev-server explicit path: build LitApp and WSGI BasicDevServer with host/port
    try:
        la = getattr(server_module, "lit_app", None)
        ws = getattr(server_module, "wsgi_serving", None)
        LitApp = getattr(la, "LitApp", None)
        BasicDevServer = getattr(ws, "BasicDevServer", None)
        if callable(LitApp) and BasicDevServer is not None:
            lit_app_obj = LitApp(models, datasets)  # type: ignore[misc]
            wsgi = getattr(lit_app_obj, "wsgi_app", None)
            if wsgi is not None:
                srv = BasicDevServer(wsgi, port=port or 5432, host=host)  # type: ignore[misc]
                # serve() takes no args per signature
                srv.serve()
                return
    except Exception:
        pass

    # Last resort: run underlying WSGI via werkzeug
    try:
        from werkzeug.serving import run_simple  # type: ignore

        # (Type alias moved to module level for static typing)

        def find_wsgi_callable(obj):
            # Try direct
            if callable(obj):
                return obj, "<app>"
            # Try common attributes and one level of nesting
            attrs = [
                "app",
                "wsgi_app",
                "application",
                "flask_app",
                "flask",
                "_app",
                "_wsgi_app",
            ]
            for a in attrs:
                val = getattr(obj, a, None)
                if callable(val):
                    return val, f"app.{a}"
                if val is not None:
                    for b in ("wsgi_app", "app", "application"):
                        sub = getattr(val, b, None)
                        if callable(sub):
                            return sub, f"app.{a}.{b}"
            return None, None

        wsgi_callable, src = find_wsgi_callable(app)
        if wsgi_callable is not None:
            print(f"[LIT] Fallback: werkzeug.run_simple using {src}")
            run_simple(
                hostname=host,
                port=port or 5432,
                application=cast(Any, wsgi_callable),
            )
            return
    except Exception:
        pass

    raise RuntimeError("Unable to start LIT server by any known method")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LIT server wrapper (robust starter)")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5432)
    parser.add_argument("--open-browser", action="store_true")
    args = parser.parse_args()
    main(args.host, args.port, args.open_browser)
