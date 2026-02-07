import asyncio
import json
import os

from dotenv import load_dotenv
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import MaxMessageTermination

from agents.summary_agent import create_agent as create_summary_agent
from agents.action_agent import create_agent as create_action_agent
from agents.risk_agent import create_agent as create_risk_agent

load_dotenv()


def get_model_client() -> OpenAIChatCompletionClient:
    """Build the model client via OpenRouter."""
    return OpenAIChatCompletionClient(
        model=os.getenv("MODEL_NAME", "arcee-ai/trinity-large-preview:free"),
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        model_info={
            "vision": False,
            "function_calling": False,
            "json_output": True,
            "structured_output": False,
            "family": "unknown",
        },
    )


def build_user_message(document_chunks: list[str], global_context: dict) -> str:
    """Format the input payload that each agent receives."""
    return json.dumps({
        "document_chunks": document_chunks,
        "global_context": global_context,
    }, indent=2)


def parse_json(text: str) -> dict:
    """Parse JSON from agent response, with fallback for wrapped responses."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end > start:
            return json.loads(text[start:end])
        return {"raw": text}


async def run_agents(document_chunks: list[str], global_context: dict | None = None) -> dict:
    """Create a RoundRobinGroupChat with all 3 agents and run them."""
    if global_context is None:
        global_context = {"entities": [], "decisions": [], "constraints": []}

    model_client = get_model_client()

    # Create the 3 specialist agents
    summary_agent = create_summary_agent(model_client)
    action_agent = create_action_agent(model_client)
    risk_agent = create_risk_agent(model_client)

    # Build RoundRobinGroupChat — each agent takes one turn
    team = RoundRobinGroupChat(
        participants=[summary_agent, action_agent, risk_agent],
        termination_condition=MaxMessageTermination(max_messages=4),
    )

    message = build_user_message(document_chunks, global_context)

    print("Running RoundRobinGroupChat with 3 agents...")
    task_result = await team.run(task=message)

    # Parse each agent's response from the chat messages
    results = {}
    for msg in task_result.messages:
        if msg.source == "Summary_Agent":
            results["summary"] = parse_json(msg.content)
        elif msg.source == "Action_Agent":
            results["actions"] = parse_json(msg.content)
        elif msg.source == "Risk_Agent":
            results["risks"] = parse_json(msg.content)

    return results


# ---- Example usage ----
if __name__ == "__main__":
    sample_chunks = [
        "The project must migrate all user data from PostgreSQL to MongoDB by Q3 2025. "
        "The backend team (led by Sarah) owns the migration pipeline. "
        "No downtime is acceptable during migration.",

        "Legal team has not yet confirmed GDPR compliance for the new storage layer. "
        "Budget for cloud infrastructure has not been finalized. "
        "The frontend team depends on the new API schema from the backend migration.",
    ]

    output = asyncio.run(run_agents(sample_chunks))
    print("\n" + "=" * 60)
    print("FINAL COMBINED OUTPUT")
    print("=" * 60)
    print(json.dumps(output, indent=2))
