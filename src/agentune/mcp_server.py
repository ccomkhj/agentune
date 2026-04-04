"""MCP server: agent-facing control plane with 5 tools."""

from __future__ import annotations

import json
import os

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from agentune.core.campaign import CampaignService
from agentune.core.db import Database
from agentune.core.models import ActionProposal


# --- Handler functions (testable without MCP transport) ---

def _get_campaign_or_raise(service: CampaignService, campaign_name: str) -> dict:
    campaign = service.get_campaign_by_name(campaign_name)
    if campaign is None:
        raise ValueError(f"Campaign '{campaign_name}' not found")
    return campaign


def handle_list_campaigns(db: Database) -> list[dict]:
    with db.connection() as conn:
        cur = conn.execute("SELECT id, name, state, metric_name, objective_direction FROM campaigns ORDER BY id")
        return cur.fetchall()


def handle_get_campaign_status(db: Database, campaign_name: str) -> dict:
    service = CampaignService(db)
    campaign = _get_campaign_or_raise(service, campaign_name)
    rounds = service.get_rounds(campaign["id"])
    latest_round = rounds[-1] if rounds else None
    return {
        **campaign,
        "total_rounds": len(rounds),
        "latest_round": latest_round,
    }


def handle_get_round_summary(db: Database, campaign_name: str, round_number: int | None = None) -> dict:
    service = CampaignService(db)
    campaign = _get_campaign_or_raise(service, campaign_name)
    rounds = service.get_rounds(campaign["id"])

    if round_number is not None:
        target_round = next((round_row for round_row in rounds if round_row["round_number"] == round_number), None)
        if target_round is None:
            raise ValueError(f"Round {round_number} not found")
        return target_round

    if not rounds:
        raise ValueError("No rounds found")
    return rounds[-1]


def handle_get_campaign_history(db: Database, campaign_name: str) -> dict:
    service = CampaignService(db)
    campaign = _get_campaign_or_raise(service, campaign_name)
    return service.get_campaign_history(campaign["id"])


def handle_run_next_round(db: Database, campaign_name: str) -> dict:
    from agentune.datasets import load_dataset
    from agentune.runner import RoundRunner

    service = CampaignService(db)
    campaign = _get_campaign_or_raise(service, campaign_name)

    if not campaign.get("dataset"):
        raise ValueError(
            f"Campaign '{campaign_name}' has no dataset configured. "
            f"This is a legacy campaign created before dataset persistence was added. "
            f"Re-create the campaign with --dataset to use run_next_round."
        )

    split, _ = load_dataset(campaign["dataset"], seed=campaign.get("split_seed", 42))
    runner = RoundRunner(db, split)
    result = runner.run_next_round(campaign["id"])
    return {
        "status": result.status,
        "round_number": result.round_number,
        "stop_reason": result.stop_reason,
    }


def handle_submit_action_proposal(db: Database, campaign_name: str, proposal_dict: dict) -> dict:
    service = CampaignService(db)
    campaign = _get_campaign_or_raise(service, campaign_name)
    proposal = ActionProposal.from_dict(proposal_dict)
    return service.submit_proposal(campaign["id"], proposal)


def handle_generate_report(db: Database, campaign_name: str) -> str:
    from agentune.report import generate_report

    return generate_report(db, campaign_name)


def handle_get_tuning_guide(backend_name: str) -> dict:
    from agentune.backends import get_backend

    backend_cls = get_backend(backend_name)
    backend = backend_cls()
    return backend.tuning_guide().to_dict()


# --- MCP Server setup ---

def create_server() -> Server:
    server = Server("agentune")

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [
            Tool(
                name="list_campaigns",
                description="List all optimization campaigns",
                inputSchema={"type": "object", "properties": {}, "required": []},
            ),
            Tool(
                name="get_campaign_status",
                description="Get current campaign state, active round, and config",
                inputSchema={
                    "type": "object",
                    "properties": {"campaign_name": {"type": "string"}},
                    "required": ["campaign_name"],
                },
            ),
            Tool(
                name="get_round_summary",
                description="Get summary for a specific round or the latest round",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "campaign_name": {"type": "string"},
                        "round_number": {"type": "integer"},
                    },
                    "required": ["campaign_name"],
                },
            ),
            Tool(
                name="get_campaign_history",
                description="Get all rounds and agent decisions for a campaign",
                inputSchema={
                    "type": "object",
                    "properties": {"campaign_name": {"type": "string"}},
                    "required": ["campaign_name"],
                },
            ),
            Tool(
                name="run_next_round",
                description="Execute the next PROPOSED round for a campaign. Runs Optuna trials, generates summary, checks stop conditions. Returns status (AWAITING_AGENT, COMPLETED, or FAILED).",
                inputSchema={
                    "type": "object",
                    "properties": {"campaign_name": {"type": "string"}},
                    "required": ["campaign_name"],
                },
            ),
            Tool(
                name="submit_action_proposal",
                description="Propose the next action for a campaign",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "campaign_name": {"type": "string"},
                        "proposal": {
                            "type": "object",
                            "properties": {
                                "action": {"type": "string", "enum": ["continue", "narrow_search", "widen_search", "increase_budget", "revise_search", "stop"]},
                                "justification": {"type": "string"},
                                "proposed_search_space": {"type": "array"},
                                "proposed_budget": {"type": "integer"},
                                "reference_round_ids": {"type": "array", "items": {"type": "integer"}},
                            },
                            "required": ["action", "justification", "reference_round_ids"],
                        },
                    },
                    "required": ["campaign_name", "proposal"],
                },
            ),
            Tool(
                name="generate_report",
                description="Generate an HTML report for a campaign with score progression, round details, decisions, and best params",
                inputSchema={
                    "type": "object",
                    "properties": {"campaign_name": {"type": "string"}},
                    "required": ["campaign_name"],
                },
            ),
            Tool(
                name="get_tuning_guide",
                description="Get backend-specific tuning knowledge: what each param does, how they interact, diagnostic patterns (overfitting/underfitting signals), and recommended tuning order. Read this BEFORE making decisions.",
                inputSchema={
                    "type": "object",
                    "properties": {"backend_name": {"type": "string", "enum": ["xgboost", "lightgbm", "catboost"]}},
                    "required": ["backend_name"],
                },
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[TextContent]:
        db = _get_db()
        try:
            if name == "list_campaigns":
                result = handle_list_campaigns(db)
            elif name == "get_campaign_status":
                result = handle_get_campaign_status(db, arguments["campaign_name"])
            elif name == "get_round_summary":
                result = handle_get_round_summary(db, arguments["campaign_name"], arguments.get("round_number"))
            elif name == "get_campaign_history":
                result = handle_get_campaign_history(db, arguments["campaign_name"])
            elif name == "run_next_round":
                result = handle_run_next_round(db, arguments["campaign_name"])
            elif name == "submit_action_proposal":
                result = handle_submit_action_proposal(db, arguments["campaign_name"], arguments["proposal"])
            elif name == "generate_report":
                html = handle_generate_report(db, arguments["campaign_name"])
                return [TextContent(type="text", text=html)]
            elif name == "get_tuning_guide":
                result = handle_get_tuning_guide(arguments["backend_name"])
            else:
                result = {"error": f"Unknown tool: {name}"}
            return [TextContent(type="text", text=json.dumps(result, default=str))]
        finally:
            db.close()

    return server


def _get_db() -> Database:
    url = os.environ.get("AGENTUNE_DB_URL", "postgresql://localhost:5432/agentune")
    db = Database(url)
    db.setup_schema()
    return db


async def main() -> None:
    server = create_server()
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
