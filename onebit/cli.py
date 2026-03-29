"""CLI interface for onebit."""

from __future__ import annotations

import logging
import sys

import click
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()


def _setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[logging.StreamHandler()],
    )


@click.group()
@click.version_option(package_name="onebit")
def cli():
    """onebit — Run 1-bit LLMs on your Mac with Metal GPU acceleration."""
    pass


@cli.command()
@click.argument("model_name")
@click.option("--prompt", "-p", default=None, help="Single prompt (non-interactive mode)")
@click.option("--max-tokens", "-n", default=512, help="Maximum tokens to generate")
@click.option("--temperature", "-t", default=0.7, help="Sampling temperature (0 = greedy)")
@click.option("--top-p", default=0.9, help="Top-p (nucleus) sampling threshold")
@click.option("--no-metal-kernel", is_flag=True, help="Disable custom Metal kernel")
@click.option("--verbose", "-v", is_flag=True, help="Verbose logging")
def run(model_name, prompt, max_tokens, temperature, top_p, no_metal_kernel, verbose):
    """Run a ternary model. Provide a registry name, HF repo, or local path.

    Examples:

        onebit run qwen2.5-3b

        onebit run phi-4-14b -p "Write a haiku about AI"

        onebit run ./my-converted-model
    """
    _setup_logging(verbose)

    from onebit.engine import load_model
    from onebit.generate import generate_stream
    from onebit.bench import get_memory_mb

    use_kernel = not no_metal_kernel

    # Load model
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task(f"Loading {model_name}...", total=None)
        try:
            model, tokenizer = load_model(model_name, use_metal_kernel=use_kernel)
        except Exception as e:
            console.print(f"[bold red]Error loading model:[/bold red] {e}")
            sys.exit(1)

    mem_mb = get_memory_mb()
    console.print(
        f"[bold green]Loaded[/bold green] {model_name} | "
        f"RAM: {mem_mb:.0f} MB | Device: Metal GPU"
    )
    console.print()

    if prompt:
        # Single-shot generation
        _generate_and_print(model, tokenizer, prompt, max_tokens, temperature, top_p)
    else:
        # Interactive chat
        console.print("[dim]Interactive mode. Type your message, Ctrl+C to exit.[/dim]")
        console.print()
        try:
            while True:
                user_input = console.input("[bold blue]You:[/bold blue] ")
                if not user_input.strip():
                    continue
                console.print("[bold green]Assistant:[/bold green] ", end="")
                _generate_and_print(
                    model, tokenizer, user_input, max_tokens, temperature, top_p
                )
                console.print()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Bye![/dim]")


def _generate_and_print(model, tokenizer, prompt, max_tokens, temperature, top_p):
    """Generate and stream text to console with live stats."""
    from onebit.generate import generate_stream

    stats = None
    for text, stats in generate_stream(
        model, tokenizer, prompt, max_tokens, temperature, top_p
    ):
        console.print(text, end="", highlight=False)

    console.print()
    if stats and stats.generated_tokens > 0:
        console.print(
            f"[dim]  {stats.generated_tokens} tokens | "
            f"prefill: {stats.prefill_tps:.1f} tok/s | "
            f"decode: {stats.decode_tps:.1f} tok/s | "
            f"total: {stats.total_time_s:.1f}s[/dim]"
        )


@cli.command()
@click.argument("model_name")
@click.option("--max-tokens", "-n", default=128, help="Tokens to generate per run")
@click.option("--runs", "-r", default=3, help="Number of benchmark runs")
@click.option("--prompt", "-p", default=None, help="Custom benchmark prompt")
@click.option("--verbose", "-v", is_flag=True, help="Verbose logging")
def bench(model_name, max_tokens, runs, prompt, verbose):
    """Benchmark a model: tok/s, memory, time-to-first-token.

    Example:

        onebit bench qwen2.5-3b --runs 3
    """
    _setup_logging(verbose)

    from onebit.engine import load_model
    from onebit.bench import benchmark_model, get_memory_mb, get_peak_memory_mb

    # Load model
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        progress.add_task(f"Loading {model_name}...", total=None)
        model, tokenizer = load_model(model_name)

    console.print(f"[bold green]Loaded[/bold green] {model_name} | RAM: {get_memory_mb():.0f} MB")
    console.print()

    # Run benchmark
    console.print(f"[bold]Running benchmark[/bold] ({runs} runs, {max_tokens} tokens each)...")
    result = benchmark_model(
        model,
        tokenizer,
        model_name=model_name,
        prompt=prompt,
        max_tokens=max_tokens,
        num_runs=runs,
    )

    if result is None:
        console.print("[red]Benchmark failed — no tokens generated.[/red]")
        sys.exit(1)

    # Display results
    console.print()
    table = Table(title=f"Benchmark: {model_name}")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    table.add_row("Prefill", f"{result.prefill_tps:.1f} tok/s")
    table.add_row("Decode", f"{result.decode_tps:.1f} tok/s")
    table.add_row("Time to First Token", f"{result.time_to_first_token_ms:.0f} ms")
    table.add_row("Peak Memory", f"{result.peak_memory_mb:.0f} MB")
    table.add_row("Prompt Tokens", str(result.prompt_tokens))
    table.add_row("Generated Tokens", str(result.generated_tokens))
    table.add_row("Total Time", f"{result.total_time_s:.2f} s")

    console.print(table)


@cli.command("convert")
@click.argument("hf_model_id")
@click.option("--output", "-o", required=True, help="Output directory for converted model")
@click.option("--keep-lm-head", is_flag=True, help="Also quantize the lm_head (not recommended)")
@click.option("--verbose", "-v", is_flag=True, help="Verbose logging")
def convert_cmd(hf_model_id, output, keep_lm_head, verbose):
    """Convert a HuggingFace model to 1.58-bit ternary format.

    Example:

        onebit convert Qwen/Qwen2.5-3B-Instruct -o ./qwen-3b-ternary
    """
    _setup_logging(verbose)

    from onebit.convert import convert_model

    console.print(f"[bold]Converting[/bold] {hf_model_id} to 1.58-bit ternary...")
    console.print()

    try:
        output_path = convert_model(
            hf_model_id,
            output,
            skip_lm_head=not keep_lm_head,
        )
        console.print(f"\n[bold green]Done![/bold green] Saved to: {output_path}")
        console.print(f"[dim]Run with: onebit run {output_path}[/dim]")
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        sys.exit(1)


@cli.command("list")
def list_cmd():
    """List available pre-configured models."""
    from onebit.models.registry import list_models

    table = Table(title="Available Models")
    table.add_column("Name", style="bold cyan")
    table.add_column("Params", justify="right")
    table.add_column("RAM", justify="right")
    table.add_column("Type")
    table.add_column("Description")

    for m in list_models():
        model_type = "Native ternary" if m.get("native_ternary") else "Converted"
        table.add_row(
            m["name"],
            m["params"],
            f"{m['ram_gb']:.1f} GB",
            model_type,
            m["description"],
        )

    console.print(table)
    console.print()
    console.print("[dim]Run any model with: onebit run <name>[/dim]")
    console.print("[dim]Or convert your own: onebit convert <hf-repo> -o ./output[/dim]")


@cli.command()
def info():
    """Show system info and onebit configuration."""
    import mlx.core as mx
    import platform
    from onebit import __version__

    table = Table(title="System Info")
    table.add_column("Property", style="bold")
    table.add_column("Value")

    table.add_row("onebit version", __version__)
    table.add_row("MLX version", mx.__version__ if hasattr(mx, "__version__") else "unknown")
    table.add_row("Python", platform.python_version())
    table.add_row("Platform", f"{platform.machine()} / {platform.system()}")
    try:
        metal_avail = mx.metal.is_available()
    except Exception:
        metal_avail = True  # MLX on Apple Silicon always has Metal
    table.add_row("Metal GPU", "Available" if metal_avail else "Not available")

    try:
        mem = mx.get_active_memory() / (1024 * 1024)
        table.add_row("GPU Memory (active)", f"{mem:.0f} MB")
    except Exception:
        pass

    console.print(table)


if __name__ == "__main__":
    cli()
