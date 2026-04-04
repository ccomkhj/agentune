"""MCP server: agent-facing control plane with 5 tools."""

from __future__ import annotations

import json
import os
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from agent_hpo.core.db import Database
from agent_hpo.core.campaign import CampaignService
from agent_hpo.core.models import ActionProposal


# --- Handler functions (testable without MCP transport) ---

def handle_list_campaigns(db: Database) -> list[dict]:
    with db.connection() as conn:
        cur = conn.execute("SELECT id, name, state, metric_name, objective_direction FROM campaigns ORDER BY id")
        return cur.fetchall()


def handle_get_campaign_status(db: Database, campaign_name: str) -> dict:
    service = CampaignService(db)
    with db.connection() as conn:
        cur = conn.execute("SELECT * FROM campaigns WHERE name = %s", (campaign_name,))
        campaign = cur.fetchone()
    if not campaign:
        raise ValueError(f"Campaign '{campaign_name}' not found")

    rounds = service.get_rounds(campaign["id"])
    latest_round = rounds[-1] if rounds else None
    return {
        **campaign,
        "total_rounds": len(rounds),
        "latest_round": latest_round,
    }


def handle_get_round_summary(db: Database, campaign_name: str, round_number: int | None = None) -> dict:
    with db.connection() as conn:
        cur = conn.execute("SELECT id FROM campaigns WHERE name = %s", (campaign_name,))
        campaign = cur.fetchone()
    if not campaign:
        raise ValueError(f"Campaign '{campaign_name}' not found")

    service = CampaignService(db)
    rounds = service.get_rounds(campaign["id"])

    if round_number is not None:
        target = [r for r in rounds if r["round_number"] == round_number]
        if not target:
            raise ValueError(f"Round {round_number} not found")
        return target[0]
    else:
        if not rounds:
            raise ValueError("No rounds found")
        return rounds[-1]


def handle_get_campaign_history(db: Database, campaign_name: str) -> dict:
    with db.connection() as conn:
        cur = conn.execute("SELECT id FROM campaigns WHERE name = %s", (campaign_name,))
        campaign = cur.fetchone()
    if not campaign:
        raise ValueError(f"Campaign '{campaign_name}' not found")

    service = CampaignService(db)
    return service.get_campaign_history(campaign["id"])


def handle_submit_action_proposal(db: Database, campaign_name: str, proposal_dict: dict) -> dict:
    with db.connection() as conn:
        cur = conn.execute("SELECT id FROM campaigns WHERE name = %s", (campaign_name,))
        campaign = cur.fetchone()
    if not campaign:
        raise ValueError(f"Campaign '{campaign_name}' not found")

    proposal = ActionProposal.from_dict(proposal_dict)
    service = CampaignService(db)
    return service.submit_proposal(campaign["id"], proposal)


# --- MCP Server setup ---

def create_server() -> Server:
    server = Server("agent-hpo")

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
                name="submit_action_proposal",
                description="Propose the next action for a campaign",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "campaign_name": {"type": "string"},
                        "proposal": {
                            "type": "object",
                            "properties": {
                                "action": {"type": "string", "enum": ["continue", "narrow_search", "widen_search", "increase_budget", "stop"]},
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
            elif name == "submit_action_proposal":
                result = handle_submit_action_proposal(db, arguments["campaign_name"], arguments["proposal"])
            else:
                result = {"error": f"Unknown tool: {name}"}
            return [TextContent(type="text", text=json.dumps(result, default=str))]
        finally:
            db.close()

    return server


def _get_db() -> Database:
    url = os.environ.get("AGENT_HPO_DB_URL", "postgresql://localhost:5432/agent_hpo")
    db = Database(url)
    db.setup_schema()
    return db


async def main():
    server = create_server()
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
