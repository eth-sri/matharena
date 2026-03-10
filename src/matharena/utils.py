import json
import os
import time
from json import tool
from re import I

import sympy
from loguru import logger


def save_run_for_recovery(came_from, original_path, solver_response, grader_response):
    ts = int(time.time())
    filename = f"logs/broken_runs/{ts}.json"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    logger.error(
        f"Error during run normalization/validation. Saving inputs to {filename} for potential recovery. Rethrowing."
    )
    # TODO: implement a recovery flow if this ever happens with a high enough rerunning cost.
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(
            {
                "came_from": came_from,
                "ts": ts,
                "original_path": original_path,
                "solver_response": {
                    "conversation": solver_response.conversation,
                    "detailed_cost": solver_response.detailed_cost,
                    "history": solver_response.history,
                },
                "grader_response": {
                    "answer": convert_answer_to_string(grader_response[0]) if grader_response else None,
                    "is_correct": grader_response[1] if grader_response else None,
                    "warning": grader_response[2] if grader_response else None,
                },
            },
            f,
            indent=4,
            ensure_ascii=False,
        )
    return


def get_substring(text, markers, mode):
    """
    Extracts a substring from text based on markers and mode.
    Args:
        text (str): The input text.
        markers (str or list of str): The marker(s) to look for.
        mode (str): "after" to get text after the last marker, "before" to get text before the first marker.
    Returns:
        str: The extracted substring.
    """

    if isinstance(markers, str):
        markers = [markers]
    for marker in markers:
        idx = text.find(marker)
        if idx == -1:
            continue
        if mode == "after":
            text = text[idx + len(marker) :].strip()
        elif mode == "before":
            text = text[:idx].strip()
        else:
            raise ValueError(f"Unknown mode '{mode}' for get_substring.")
    return text


def check_for_extra_keys(d, allowed_keys):
    for k in d.keys():
        if k not in allowed_keys:
            raise ValueError(f"Unexpected key {k} in message: {d}")


def is_conversation_broken(msg_list):
    """
    Checks if a list of messages is obviously broken (no last nonempty assistant message).
    Usually this means this should be deleted and retried.
    """
    if len(msg_list) == 0:
        return True, "Empty message list"
    last_msg = msg_list[-1]
    if last_msg["role"] != "assistant":
        return True, "Last message is not from assistant"
    if last_msg.get("type", "response") != "response":
        return True, "Last message is not a response"
    return False, ""


def normalize_conversation(messages):
    """
    Utility function for converting a conversation to the normalized/clean (.json) format from any other possible format.

    CleanMessages: [CleanMessage]
    CleanMessage: {role: Role}
    Role and its extra keys:
        "developer" (ex "system"): {content: str}
        "user": {content: str}
        "tool_response": {tool_name: str, tool_call_id: Optional[str]}
        "assistant": {type: Type, ...}
            Type and its extra keys:
                "cot": {content: str}
                "response": {content: str}
                "tool_call": {tool_name: str, tool_call_id: Optional[str], arguments: dict}
                "internal_tool_call":
                    only "code_interpreter" for now: {tool_name: str, code: str}

    """
    clean_messages = []

    pending_tool_calls = 0
    for m in messages:
        cm = {}
        if "role" not in m:
            role = "assistant"
            if m.get("type", "") != "reasoning":
                logger.warning(
                    f"Message missing role, defaulting to 'assistant'. This is fully expected for GPT models with response API on Euler with tool use.\n"
                )
        else:
            role = m["role"]

        # Simple roles
        if role in ["user", "system", "developer"]:
            cm["role"] = "developer" if role == "system" else role
            cm["content"] = m.get("content", m.get("output", ""))
            check_for_extra_keys(m, ["role", "content"])
            clean_messages.append(cm)
            continue

        # Tool response
        if role in ["tool", "function_call_output", "tool_response"] or m.get("type", None) == "function_call_output":
            cm["role"] = "tool_response"
            cm["content"] = m.get("content", m.get("output", ""))

            # There has to be a tool call this replies to
            if pending_tool_calls <= 0:
                logger.warning(f"Tool response without a pending tool call at: {m}")
            pending_tool_calls -= 1

            # Get tool name
            if "tool_name" in m or "name" in m:
                cm["tool_name"] = m.get("tool_name", m.get("name"))
            else:
                logger.warning(f"Tool response missing tool_name, defaulting to execute_code")
                cm["tool_name"] = "execute_code"

            # Get tool call id
            if "tool_call_id" in m or "id" in m or "call_id" in m:
                cm["tool_call_id"] = m.get("tool_call_id", m.get("id", m.get("call_id")))
            else:
                cm["tool_call_id"] = None
            check_for_extra_keys(m, ["role", "type", "content", "tool_name", "tool_call_id", "id", "call_id", "name", "output"])
            clean_messages.append(cm)
            continue

        # Internal tool call
        # NOTE: we only expect code_interpreter for now, tweak code here if using more
        if role == "code-internal" or (
            role == "assistant" and m.get("type", None) in ["internal_tool_call", "code_interpreter_call", "web_search_call"]
        ):
            cm["role"] = "assistant"
            cm["type"] = "internal_tool_call"
            cm["tool_name"] = "code_interpreter" if m.get("type", None) in ["internal_tool_call", "code_interpreter_call"] else "web_search"
            if "web_search_call" == cm["tool_name"]:
                cm["content"] = m.get("query", "")
            else:
                cm["code"] = m.get("code", m.get("content", None))
            check_for_extra_keys(m, ["role", "content", "type", "id", "container_id", "tool_name", "code", "outputs", "query"])
            clean_messages.append(cm)
            continue

        # External tool call
        if role in ["function", "function_call", "code"] or (
            role == "assistant" and m.get("type", None) in ["tool_call", "function_call"]
        ):
            cm["role"] = "assistant"
            cm["type"] = "tool_call"

            # Sometimes it's a json string with arguments
            if "content" in m:
                try:
                    toolcall_dict = json.loads(m["content"]) if m.get("content", None) is not None else None
                except json.JSONDecodeError:
                    if "tool_arguments" in m["content"]:
                        logger.warning(f"Should have probably been able to parse as JSON...")
                    toolcall_dict = None

            # Get tool name
            if "tool_name" in m or "name" in m:
                cm["tool_name"] = m.get("tool_name", m.get("name"))
            elif role == "code" and toolcall_dict is not None and "tool_name" in toolcall_dict:
                cm["tool_name"] = toolcall_dict["tool_name"]
            else:
                raise RuntimeError(f"Tool call missing tool_name: {m}")

            # Get tool call id which is optional
            if "tool_call_id" in m or "id" in m or "call_id" in m:
                cm["tool_call_id"] = m.get("tool_call_id", m.get("id", m.get("call_id")))
            else:
                cm["tool_call_id"] = None

            # Get arguments
            if "arguments" in m:
                if isinstance(m["arguments"], str):
                    try:
                        cm["arguments"] = json.loads(m["arguments"])
                    except json.JSONDecodeError:
                        logger.warning(f"Could not parse tool call arguments as JSON, defaulting to raw string")
                        cm["arguments"] = m["arguments"]
                else:
                    cm["arguments"] = m["arguments"]
            elif toolcall_dict is not None and "tool_arguments" in toolcall_dict:
                cm["arguments"] = toolcall_dict["tool_arguments"]
            elif toolcall_dict is not None and "lang" in toolcall_dict:
                cm["arguments"] = {"code": toolcall_dict.get("code", ""), "lang": toolcall_dict.get("lang", "python")}
            else:
                cm["arguments"] = {"code": m["content"]}  # for backward compatibility
                logger.warning(f"Tool call missing arguments at: {m}, defaulting to just code")

            check_for_extra_keys(
                m, ["role", "content", "type", "tool_name", "name", "tool_call_id", "id", "call_id", "arguments"]
            )

            clean_messages.append(cm)
            pending_tool_calls += 1
            continue

        # Rest of assistant types
        assert role == "assistant", f"Unknown role: {m['role']}"
        cm["role"] = "assistant"
        cm["content"] = m.get("content", m.get("output", ""))
        if m.get("type", None) in ["cot", "thinking", "reasoning"]:
            if (cm["content"] is None or len(cm["content"]) == 0) and m.get("summary", None) is not None:
                # "Summary" OpenAI case so patch the content from summary blocks
                summary = ""
                for thought in m["summary"]:
                    if thought["text"] is not None:
                        summary += "<thought>" + "\n" + thought["text"] + "\n" + "</thought>\n"
                cm["content"] = summary
            cm["type"] = "cot"
            clean_messages.append(cm)
        elif "type" not in m or m["type"] == "response":
            cm["type"] = "response"
            has_cot = False
            if "<thought>" in cm["content"] or "</thought>" in cm["content"]:
                cm["content"] = cm["content"].replace("<thought>", "<think>").replace("</thought>", "</think>")
            if "<think>" in cm["content"] or "</think>" in cm["content"]:
                cot_tag = "think"
                has_cot = True
            
            if has_cot:
                cot_start = cm["content"].find(f"<{cot_tag}>")
                if cot_start == -1:
                    cot_start = 0  # Some of our traces have only the end tag
                cot_end = cm["content"].rfind(f"</{cot_tag}>") + len(f"</{cot_tag}>")
                thinking_message = {
                    "role": "assistant",
                    "type": "cot",
                    "content": cm["content"][cot_start:cot_end].strip(),
                }
                clean_messages.append(thinking_message)
                cm["content"] = (cm["content"][:cot_start] + cm["content"][cot_end:]).strip()
                # assert (
                #     "think>" not in cm["content"] and "thought>" not in cm["content"]
                # ), f"Multiple CoT tags in message: {m}"
            clean_messages.append(cm)
        else:
            raise ValueError(f"Unknown assistant type: {m['type']}")

        # Were there also additional tool calls?
        if "tool_calls" in m and m["tool_calls"] is not None and len(m["tool_calls"]) > 0:
            for tc in m["tool_calls"]:
                cm = {}
                cm["role"] = "assistant"
                cm["type"] = "tool_call"
                cm["content"] = ""  # tool calls have no content
                cm["tool_name"] = tc["function"]["name"]
                cm["tool_call_id"] = tc["id"]
                cm["arguments"] = tc["function"]["arguments"]
                clean_messages.append(cm)
                check_for_extra_keys(tc, ["function", "extra_content", "id", "type", "index"])
                pending_tool_calls += 1
        ignored_keys = [
            "refusal",
            "annotations",
            "function_call",
            "audio",
            "id",
            "extra_content",
            "reasoning_details",
            "reasoning_content",
            "encrypted_content",
            "status",
            "summary",
            "thought_signature",
        ]
        all_keys = ["role", "content", "type", "tool_calls"] + ignored_keys
        check_for_extra_keys(m, all_keys)
    return clean_messages


def safe_str_int(x, max_digits=4300):
    """Converts an integer to a string, handling large integers.

    Args:
        x: The integer to convert.
        max_digits (int, optional): The maximum number of digits to display. Defaults to 4300.

    Returns:
        str: The string representation of the integer.
    """
    s = str(x)
    if len(s) > max_digits:
        return f"{s[:20]}...({len(s)} digits)...{s[-20:]}"
    return s


def convert_answer_to_string(answer):
    """Converts an answer to a string.

    Args:
        answer: The answer to convert.

    Returns:
        str: The string representation of the answer.
    """
    try:
        if isinstance(answer, sympy.Integer):
            return safe_str_int(int(answer))
        else:
            return safe_str_int(answer)
    except Exception as e:  # noqa: E722
        logger.warning(f"Exception when converting answer to string: {e}")
        return "None"


def lists_differ(l1, l2):
    if l1 is None and l2 is not None:
        return True
    if l1 is not None and l2 is None:
        return True
    if l1 is None and l2 is None:
        return False
    if len(l1) != len(l2):
        return True
    for i in range(len(l1)):
        if l1[i] != l2[i]:
            return True
    return False
