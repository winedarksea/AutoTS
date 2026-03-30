"""MCP Prompt & Resource definitions for the AutoTS MCP server."""

import json
from os.path import dirname, join, exists

try:
    from mcp.types import (
        Prompt,
        PromptArgument,
        PromptMessage,
        GetPromptResult,
        TextContent,
        Resource,
    )

    def get_resources(mcp_file_path: str) -> list[Resource]:
        """Return list of available documentation resources."""
        resources = []
        base_path = dirname(dirname(dirname(mcp_file_path)))
        mcp_path = dirname(mcp_file_path)

        doc_files = [
            ("README.md", "MCP Server README", mcp_path),
            ("extended_tutorial.md", "Extended AutoTS Tutorial", base_path),
            ("production_example.py", "Production Example Script", base_path),
            ("README.md", "AutoTS Main README", base_path),
            ("TODO.md", "AutoTS Development Roadmap", base_path),
        ]

        for filename, description, base in doc_files:
            filepath = join(base, filename)
            if exists(filepath):
                resources.append(
                    Resource(
                        uri=f"file://{filepath}",
                        name=filename,
                        description=description,
                        mimeType=(
                            "text/markdown"
                            if filename.endswith('.md')
                            else "text/plain"
                        ),
                    )
                )

        resources.append(
            Resource(
                uri="autots://docs/forecast_custom_params",
                name="AutoTS forecast_custom Parameters",
                description="Parameter reference for the forecast_custom tool — read this before using forecast_custom",
                mimeType="application/json",
            )
        )

        return resources

    AUTOTS_DOCS = json.dumps(
        {
            "AutoTS_Parameters": {
                "forecast_length": "Number of periods to forecast (required)",
                "frequency": "Pandas frequency string ('D','H','W','MS',etc.) or 'infer'",
                "ensemble": "Ensemble method: 'simple','distance','horizontal','mosaic',None",
                "model_list": "List of models or preset: 'fast','superfast','scalable','all','default'",
                "transformer_list": "Transformations: 'fast','superfast','all'",
                "max_generations": "Number of genetic algorithm generations, generally controls runtime.",
                "generation_timeout": "Max time (minutes) for all generations. Useful to set a sanity cap on runtime.",
                "num_validations": "Number of cross-validation splits",
                "validation_method": "'backwards','even','seasonal',etc.",
                "models_to_validate": "Fraction of models to fully validate (0.0-1.0)",
                "n_jobs": "Parallel processes: 'auto',-1,or specific number",
            },
            "Example": {
                "autots_params": {
                    "forecast_length": 30,
                    "frequency": "D",
                    "ensemble": "simple",
                    "model_list": "fast",
                    "max_generations": 5,
                }
            },
            "Documentation": "See extended_tutorial.md for complete documentation",
        },
        indent=2,
    )

    async def read_resource(uri: str) -> str:
        """Read a documentation resource by URI."""
        if uri == "autots://docs/forecast_custom_params":
            return AUTOTS_DOCS
        elif uri.startswith("file://"):
            filepath = uri[7:]
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    return f.read()
            except Exception as e:
                return f"Error reading {filepath}: {str(e)}"
        else:
            return f"Unknown resource URI: {uri}"

    PROMPTS = [
        Prompt(
            name="sample_forecast",
            title="Sample Forecast Workflow",
            description="Load a built-in sample dataset, run a fast mosaic ensemble forecast (4 weeks), and plot the results.",
            arguments=[
                PromptArgument(
                    name="dataset",
                    description="Sample dataset to use: daily, hourly, weekly, monthly, yearly, linear, sine, artificial",
                    required=False,
                ),
            ],
        ),
        Prompt(
            name="explainable_forecast",
            title="Explainable Forecast from File",
            description="Load data from a CSV file, run an explainable model forecast, then return forecast components and validation results.",
            arguments=[
                PromptArgument(
                    name="filepath",
                    description="Path to a CSV file with a datetime index column",
                    required=True,
                ),
                PromptArgument(
                    name="forecast_length",
                    description="Number of periods to forecast (default: 30)",
                    required=False,
                ),
            ],
        ),
    ]

    async def get_prompt(name: str, arguments: dict | None = None) -> GetPromptResult:
        """Return a multi-step workflow prompt by name."""
        arguments = arguments or {}

        if name == "sample_forecast":
            dataset = arguments.get("dataset", "daily")
            return GetPromptResult(
                description=f"Fast forecast workflow using the '{dataset}' sample dataset",
                messages=[
                    PromptMessage(
                        role="user",
                        content=TextContent(
                            type="text",
                            text=(
                                f"Run a complete sample forecast workflow:\n"
                                f"1. Call load_sample_data with dataset=\"{dataset}\"\n"
                                f"2. Using the returned data_id, call forecast_fast with forecast_length=28 (4 weeks)\n"
                                f"3. Using the returned prediction_id, call plot_forecast with plot_all=true\n"
                                f"Return the plot and a brief summary of the forecast."
                            ),
                        ),
                    ),
                ],
            )

        elif name == "explainable_forecast":
            filepath = arguments.get("filepath", "")
            forecast_length = arguments.get("forecast_length", "30")
            if not filepath:
                return GetPromptResult(
                    description="Error: filepath argument is required",
                    messages=[
                        PromptMessage(
                            role="user",
                            content=TextContent(
                                type="text",
                                text="Error: the 'filepath' argument is required for the explainable_forecast prompt.",
                            ),
                        ),
                    ],
                )
            return GetPromptResult(
                description=f"Explainable forecast workflow for {filepath}",
                messages=[
                    PromptMessage(
                        role="user",
                        content=TextContent(
                            type="text",
                            text=(
                                f"Run a complete explainable forecast workflow:\n"
                                f"1. Call load_data_from_file with filepath=\"{filepath}\"\n"
                                f"2. Using the returned data_id, call forecast_explainable with forecast_length={forecast_length}\n"
                                f"3. Using the returned prediction_id, call get_forecast_components to get the decomposition\n"
                                f"4. Using the returned autots_id, call get_validation_results to get model rankings\n"
                                f"Return the component decomposition, validation results, and a summary of the best model."
                            ),
                        ),
                    ),
                ],
            )

        raise ValueError(f"Unknown prompt: {name}")

except ImportError:
    PROMPTS = []
    AUTOTS_DOCS = "{}"

    def get_resources(mcp_file_path: str) -> list:
        return []

    async def read_resource(uri: str) -> str:
        return f"MCP not available"

    async def get_prompt(name: str, arguments: dict | None = None):
        raise ValueError(f"MCP not available")
