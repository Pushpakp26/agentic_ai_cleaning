import json
from datetime import datetime
from typing import AsyncGenerator, Dict, Optional


def format_sse(data: str, event: Optional[str] = None, id: Optional[str] = None) -> str:
	"""Format a Server-Sent Event string."""
	lines = []
	if event:
		lines.append(f"event: {event}")
	if id:
		lines.append(f"id: {id}")
	for line in data.split("\n"):
		lines.append(f"data: {line}")
	return "\n".join(lines) + "\n\n"


async def stream_progress(messages: AsyncGenerator[Dict, None]):
	"""Yield SSE-formatted messages from an async generator of dicts.

	Each message should include keys: {"type": str, "message": str, "progress": int}
	"""
	last_id = 0
	async for msg in messages:
		last_id += 1
		payload = {
			"timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
			**msg,
		}
		# Convert to JSON string for proper parsing on client side
		yield format_sse(data=json.dumps(payload), event=msg.get("type"), id=str(last_id))


