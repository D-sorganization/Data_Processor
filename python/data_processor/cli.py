"""
Command line interface for programmatic access to the Data Processor.

The CLI intentionally focuses on two core workflows that users routinely
automate:

1. Quickly inspect a batch of files to discover available signals.
2. Run a lightweight processing pipeline (load → optional filtering →
   optional signal selection → export) defined either via CLI flags or a
   declarative JSON config file.

Example JSON pipeline:
{
  "files": ["./data/example.csv"],
  "combine": true,
  "selected_signals": ["time", "pressure", "temperature"],
  "filter": {
    "filter_type": "Moving Average",
    "ma_window": 5
  },
  "output": {
    "path": "./output/processed.csv",
    "format": "csv"
  }
}

Run with:
    python -m data_processor.cli run --config pipeline.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import typer
from rich.console import Console
from rich.table import Table

from .core.data_loader import DataLoader
from .core.signal_processor import SignalProcessor
from .logging_config import init_default_logging
from .models.processing_config import FilterConfig

console = Console()
app = typer.Typer(help="Data Processor CLI for automated workflows.")
init_default_logging()


def _load_config(config_path: Path) -> Dict[str, Any]:
    """Load a JSON pipeline configuration."""
    try:
        with config_path.open("r", encoding="utf-8") as fp:
            return json.load(fp)
    except json.JSONDecodeError as exc:
        raise typer.BadParameter(
            f"Invalid JSON in config '{config_path}': {exc}"
        ) from exc
    except OSError as exc:
        raise typer.BadParameter(f"Unable to read config '{config_path}': {exc}") from exc


def _apply_filter_if_requested(
    df,
    filter_section: Optional[Dict[str, Any]],
    signal_processor: SignalProcessor,
) -> Any:
    """Apply configured filter if specified."""
    if not filter_section:
        return df

    filter_config = FilterConfig(**filter_section)
    filtered = signal_processor.apply_filter(df, filter_config)
    return filtered


@app.command()
def detect(
    files: List[Path] = typer.Argument(..., help="One or more CSV/Parquet data files."),
    high_perf: bool = typer.Option(
        True, "--high-perf/--no-high-perf", help="Use the high performance loader."
    ),
) -> None:
    """Detect and print unique signal names from the supplied files."""
    loader = DataLoader(use_high_performance=high_perf)
    file_paths = [str(path) for path in files]

    console.rule("Signal Detection")
    signals = loader.detect_signals(file_paths)
    if not signals:
        console.print("[yellow]No signals detected.[/yellow]")
        raise typer.Exit(code=0)

    table = Table(title="Detected Signals")
    table.add_column("Signal", justify="left")
    for signal in sorted(signals):
        table.add_row(signal)

    console.print(table)


@app.command()
def run(
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to pipeline JSON config. CLI options override values inside.",
    ),
    files: Optional[List[Path]] = typer.Option(
        None,
        "--file",
        "-f",
        help="Input files (ignored when provided via config). May be repeated.",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Destination file path when not specified in config.",
    ),
    output_format: str = typer.Option(
        "csv", "--format", "-t", help="Output format fallback (csv/excel/parquet/json)."
    ),
    combine: Optional[bool] = typer.Option(
        None,
        "--combine/--no-combine",
        help="Override combine flag from config.",
    ),
    high_perf: bool = typer.Option(
        True, "--high-perf/--no-high-perf", help="Use the high performance loader."
    ),
) -> None:
    """Execute a lightweight processing pipeline."""
    pipeline: Dict[str, Any] = {}
    if config:
        pipeline.update(_load_config(config))

    if files:
        pipeline["files"] = [str(path) for path in files]

    if combine is not None:
        pipeline["combine"] = combine

    if output:
        pipeline.setdefault("output", {})
        pipeline["output"]["path"] = str(output)
        pipeline["output"]["format"] = output_format

    file_list = pipeline.get("files", [])
    if not file_list:
        raise typer.BadParameter("No input files provided. Use --file or supply a config.")

    loader = DataLoader(use_high_performance=high_perf)
    processor = SignalProcessor()

    combine_frames = pipeline.get("combine", True)
    console.rule("Loading data")
    data = loader.load_multiple_files(file_list, combine=combine_frames)

    if combine_frames:
        dataframe = data
    else:
        # Merge lazily with outer join so downstream filtering still works
        dataframe = loader.combine_dataframes(list(data.values()))

    selected_signals = pipeline.get("selected_signals")
    if selected_signals:
        missing = [col for col in selected_signals if col not in dataframe.columns]
        if missing:
            console.print(
                f"[yellow]Warning: missing signals skipped -> {', '.join(missing)}[/yellow]"
            )
        dataframe = dataframe[selected_signals]

    dataframe = _apply_filter_if_requested(
        dataframe, pipeline.get("filter"), processor
    )

    output_section = pipeline.get("output")
    if output_section:
        output_path = Path(output_section["path"]).expanduser()
        output_format = output_section.get("format", output_format)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        loader.save_dataframe(dataframe, str(output_path), format_type=output_format)
        console.print(f"[green]Saved processed data to {output_path}[/green]")
    else:
        console.print("[cyan]Pipeline completed with no output target specified.[/cyan]")


def main() -> None:
    """Entry point for `python -m data_processor.cli`."""
    app()


if __name__ == "__main__":
    main()
