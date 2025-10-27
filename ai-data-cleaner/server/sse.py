import json
import asyncio
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
	try:
		async for msg in messages:
			last_id += 1
			payload = {
				"timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
				**msg,
			}
			# Convert to JSON string for proper parsing on client side
			# IMPORTANT: Do not set a custom 'event' so browsers dispatch the default 'message' event
			sse_message = format_sse(data=json.dumps(payload), id=str(last_id))
			print(f"[SSE] Sending message type={msg.get('type')}, id={last_id}")  # Debug log
			print(f"[SSE] Message content: {sse_message[:200]}...")  # Debug log
			yield sse_message
			
			# Force flush for completion messages
			if msg.get("type") in ["complete", "error"]:
				print(f"[SSE] Completion message sent, waiting before ending stream")
				# Add small delay to ensure message reaches client
				await asyncio.sleep(1.0)
				print(f"[SSE] Ending stream after delay")
				break
			

	except Exception as e:
		print(f"[SSE] Error in stream_progress: {e}")  # Debug log
		raise
	finally:
		print(f"[SSE] Stream ended, total messages: {last_id}")


