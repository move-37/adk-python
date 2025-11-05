# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for Anthropic Extended Thinking configuration."""

import os
from unittest import mock

from anthropic import NOT_GIVEN
from anthropic import types as anthropic_types
from google.adk.models.anthropic_llm import Claude
from google.adk.models.anthropic_llm import content_block_to_part
from google.adk.models.llm_request import LlmRequest
from google.genai import types
import pytest


@pytest.fixture
def claude_llm():
  return Claude(model="claude-opus-4-1@20250805", max_tokens=4096)


@pytest.fixture
def base_llm_request():
  return LlmRequest(
      model="claude-opus-4-1@20250805",
      contents=[
          types.Content(
              role="user", parts=[types.Part.from_text(text="What is 2+2?")]
          )
      ],
      config=types.GenerateContentConfig(
          system_instruction="You are a helpful assistant"
      ),
  )


@pytest.mark.asyncio
async def test_thinking_budget_automatic_raises_error(
    claude_llm, base_llm_request
):
  """Test that thinking_budget=-1 raises ValueError."""
  base_llm_request.config.thinking_config = types.ThinkingConfig(
      include_thoughts=True, thinking_budget=-1
  )

  with pytest.raises(
      ValueError,
      match="Unlimited thinking budget \\(-1\\) is not supported with Claude",
  ):
    async for response in claude_llm.generate_content_async(
        base_llm_request, stream=False
    ):
      pass


@pytest.mark.asyncio
async def test_thinking_budget_disabled(claude_llm, base_llm_request):
  """Test that thinking_budget=0 disables thinking."""
  base_llm_request.config.thinking_config = types.ThinkingConfig(
      include_thoughts=True, thinking_budget=0
  )

  with mock.patch.object(claude_llm, "_anthropic_client") as mock_client:
    mock_client.messages.create = mock.AsyncMock(
        return_value=anthropic_types.Message(
            id="test",
            content=[anthropic_types.TextBlock(text="4", type="text")],
            model="claude-opus-4-1@20250805",
            role="assistant",
            stop_reason="end_turn",
            type="message",
            usage=anthropic_types.Usage(input_tokens=10, output_tokens=2),
        )
    )

    responses = []
    async for response in claude_llm.generate_content_async(
        base_llm_request, stream=False
    ):
      responses.append(response)

    # Verify non-streaming mode was used (thinking disabled)
    mock_client.messages.create.assert_called_once()
    # Thinking should be NOT_GIVEN when disabled
    call_kwargs = mock_client.messages.create.call_args.kwargs
    assert call_kwargs.get("thinking") == NOT_GIVEN


@pytest.mark.asyncio
async def test_thinking_budget_specific_value(claude_llm, base_llm_request):
  """Test that thinking_budget with specific value is used."""
  base_llm_request.config.thinking_config = types.ThinkingConfig(
      include_thoughts=True, thinking_budget=5000
  )

  with mock.patch.object(claude_llm, "_anthropic_client") as mock_client:
    mock_client.messages.create = mock.AsyncMock(
        return_value=anthropic_types.Message(
            id="test",
            content=[],
            model="claude-opus-4-1@20250805",
            role="assistant",
            stop_reason="end_turn",
            type="message",
            usage=anthropic_types.Usage(input_tokens=10, output_tokens=20),
        )
    )

    responses = []
    async for response in claude_llm.generate_content_async(
        base_llm_request, stream=False
    ):
      responses.append(response)

    # Verify thinking parameter was passed with 5000 tokens
    mock_client.messages.create.assert_called_once()
    call_kwargs = mock_client.messages.create.call_args.kwargs
    assert call_kwargs["thinking"] == {"type": "enabled", "budget_tokens": 5000}


@pytest.mark.asyncio
async def test_thinking_budget_minimum_value(claude_llm, base_llm_request):
  """Test that thinking budget of 1024 tokens (minimum) works."""
  base_llm_request.config.thinking_config = types.ThinkingConfig(
      include_thoughts=True, thinking_budget=1024
  )

  with mock.patch.object(claude_llm, "_anthropic_client") as mock_client:
    mock_client.messages.create = mock.AsyncMock(
        return_value=anthropic_types.Message(
            id="test",
            content=[],
            model="claude-opus-4-1@20250805",
            role="assistant",
            stop_reason="end_turn",
            type="message",
            usage=anthropic_types.Usage(input_tokens=10, output_tokens=20),
        )
    )

    responses = []
    async for response in claude_llm.generate_content_async(
        base_llm_request, stream=False
    ):
      responses.append(response)

    # Verify thinking parameter was passed with 1024 tokens
    mock_client.messages.create.assert_called_once()
    call_kwargs = mock_client.messages.create.call_args.kwargs
    assert call_kwargs["thinking"] == {"type": "enabled", "budget_tokens": 1024}


def test_thinking_block_parsing():
  """Test that thinking blocks are parsed as Part(thought=True)."""
  import base64

  # Create a mock thinking block with 'thinking' attribute
  class ThinkingBlock:

    def __init__(self):
      self.thinking = "Let me think about this step by step..."
      # Signature must be base64 encoded
      self.signature = base64.b64encode(b"mock_signature_123").decode("utf-8")

  thinking_block = ThinkingBlock()
  part = content_block_to_part(thinking_block)

  assert part.text == "Let me think about this step by step..."
  assert part.thought is True
  # Signature is stored as bytes (decoded from base64 input)
  assert part.thought_signature == b"mock_signature_123"


def test_thinking_block_type_check():
  """Test that thinking blocks with type attribute are parsed correctly."""
  import base64

  # Create a mock thinking block with 'type' attribute
  class ThinkingBlockWithType:

    def __init__(self):
      self.type = "thinking"
      # Signature must be base64 encoded
      self.signature = base64.b64encode(b"mock_signature_456").decode("utf-8")

    def __str__(self):
      return "Thinking content via type check"

  thinking_block = ThinkingBlockWithType()
  part = content_block_to_part(thinking_block)

  assert part.text == "Thinking content via type check"
  assert part.thought is True
  # Signature is stored as bytes (decoded from base64 input)
  assert part.thought_signature == b"mock_signature_456"


@pytest.mark.asyncio
async def test_no_thinking_config(claude_llm, base_llm_request):
  """Test that requests without thinking_config work normally."""
  # No thinking_config set

  with mock.patch.object(claude_llm, "_anthropic_client") as mock_client:
    mock_client.messages.create = mock.AsyncMock(
        return_value=anthropic_types.Message(
            id="test",
            content=[anthropic_types.TextBlock(text="4", type="text")],
            model="claude-opus-4-1@20250805",
            role="assistant",
            stop_reason="end_turn",
            type="message",
            usage=anthropic_types.Usage(input_tokens=10, output_tokens=2),
        )
    )

    responses = []
    async for response in claude_llm.generate_content_async(
        base_llm_request, stream=False
    ):
      responses.append(response)

    # Verify non-streaming mode was used without thinking
    mock_client.messages.create.assert_called_once()
    call_kwargs = mock_client.messages.create.call_args.kwargs
    assert call_kwargs.get("thinking") == NOT_GIVEN


@pytest.mark.asyncio
async def test_thinking_with_streaming(claude_llm, base_llm_request):
  """Test that thinking works in streaming mode."""
  base_llm_request.config.thinking_config = types.ThinkingConfig(
      include_thoughts=True, thinking_budget=2048
  )

  with mock.patch.object(claude_llm, "_anthropic_client") as mock_client:
    mock_stream = mock.AsyncMock()
    mock_stream.__aenter__ = mock.AsyncMock(return_value=mock_stream)
    mock_stream.__aexit__ = mock.AsyncMock()
    mock_stream.__aiter__ = mock.AsyncMock(return_value=iter([]))
    mock_stream.get_final_message = mock.AsyncMock(
        return_value=anthropic_types.Message(
            id="test",
            content=[],
            model="claude-opus-4-1@20250805",
            role="assistant",
            stop_reason="end_turn",
            type="message",
            usage=anthropic_types.Usage(input_tokens=10, output_tokens=20),
        )
    )
    mock_client.messages.stream = mock.Mock(return_value=mock_stream)

    responses = []
    async for response in claude_llm.generate_content_async(
        base_llm_request, stream=True
    ):
      responses.append(response)

    # Verify streaming mode was used with thinking
    mock_client.messages.stream.assert_called_once()
    call_kwargs = mock_client.messages.stream.call_args.kwargs
    assert call_kwargs["thinking"] == {"type": "enabled", "budget_tokens": 2048}


@pytest.mark.asyncio
async def test_interleaved_thinking_disabled_by_default(
    claude_llm, base_llm_request
):
  """Test that beta header is NOT sent when interleaved thinking is disabled."""
  base_llm_request.config.thinking_config = types.ThinkingConfig(
      include_thoughts=True, thinking_budget=2048
  )

  with mock.patch.object(claude_llm, "_anthropic_client") as mock_client:
    mock_client.messages.create = mock.AsyncMock(
        return_value=anthropic_types.Message(
            id="test",
            content=[],
            model="claude-opus-4-1@20250805",
            role="assistant",
            stop_reason="end_turn",
            type="message",
            usage=anthropic_types.Usage(input_tokens=10, output_tokens=20),
        )
    )

    responses = []
    async for response in claude_llm.generate_content_async(
        base_llm_request, stream=False
    ):
      responses.append(response)

    # Verify beta header is NOT_GIVEN (default behavior)
    mock_client.messages.create.assert_called_once()
    call_kwargs = mock_client.messages.create.call_args.kwargs
    assert call_kwargs.get("extra_headers") == NOT_GIVEN


@pytest.mark.asyncio
async def test_interleaved_thinking_streaming(base_llm_request):
  """Test that beta header is sent in streaming mode when provided."""
  # Create Claude with beta headers
  claude_llm = Claude(
      model="claude-opus-4-1@20250805",
      max_tokens=4096,
      extra_headers={"anthropic-beta": "interleaved-thinking-2025-05-14"},
  )

  base_llm_request.config.thinking_config = types.ThinkingConfig(
      include_thoughts=True, thinking_budget=2048
  )

  with mock.patch.object(claude_llm, "_anthropic_client") as mock_client:
    mock_stream = mock.AsyncMock()
    mock_stream.__aenter__ = mock.AsyncMock(return_value=mock_stream)
    mock_stream.__aexit__ = mock.AsyncMock()
    mock_stream.__aiter__ = mock.AsyncMock(return_value=iter([]))
    mock_stream.get_final_message = mock.AsyncMock(
        return_value=anthropic_types.Message(
            id="test",
            content=[],
            model="claude-opus-4-1@20250805",
            role="assistant",
            stop_reason="end_turn",
            type="message",
            usage=anthropic_types.Usage(input_tokens=10, output_tokens=20),
        )
    )
    mock_client.messages.stream = mock.Mock(return_value=mock_stream)

    responses = []
    async for response in claude_llm.generate_content_async(
        base_llm_request, stream=True
    ):
      responses.append(response)

    # Verify beta header was sent
    mock_client.messages.stream.assert_called_once()
    call_kwargs = mock_client.messages.stream.call_args.kwargs
    assert call_kwargs["extra_headers"] == {
        "anthropic-beta": "interleaved-thinking-2025-05-14"
    }


@pytest.mark.asyncio
async def test_interleaved_thinking_non_streaming(base_llm_request):
  """Test that beta header is sent in non-streaming mode when provided."""
  # Create Claude with beta headers
  claude_llm = Claude(
      model="claude-opus-4-1@20250805",
      max_tokens=4096,
      extra_headers={"anthropic-beta": "interleaved-thinking-2025-05-14"},
  )

  base_llm_request.config.thinking_config = types.ThinkingConfig(
      include_thoughts=True, thinking_budget=2048
  )

  with mock.patch.object(claude_llm, "_anthropic_client") as mock_client:
    mock_client.messages.create = mock.AsyncMock(
        return_value=anthropic_types.Message(
            id="test",
            content=[],
            model="claude-opus-4-1@20250805",
            role="assistant",
            stop_reason="end_turn",
            type="message",
            usage=anthropic_types.Usage(input_tokens=10, output_tokens=20),
        )
    )

    responses = []
    async for response in claude_llm.generate_content_async(
        base_llm_request, stream=False
    ):
      responses.append(response)

    # Verify beta header was sent
    mock_client.messages.create.assert_called_once()
    call_kwargs = mock_client.messages.create.call_args.kwargs
    assert call_kwargs["extra_headers"] == {
        "anthropic-beta": "interleaved-thinking-2025-05-14"
    }


@pytest.mark.asyncio
async def test_extra_headers_sent_regardless_of_thinking(base_llm_request):
  """Test that extra headers are sent even when thinking is disabled."""
  # Create Claude with beta headers
  claude_llm = Claude(
      model="claude-opus-4-1@20250805",
      max_tokens=4096,
      extra_headers={"anthropic-beta": "interleaved-thinking-2025-05-14"},
  )

  # No thinking_config set - thinking disabled
  base_llm_request.config.thinking_config = types.ThinkingConfig(
      include_thoughts=True, thinking_budget=0  # Disabled
  )

  with mock.patch.object(claude_llm, "_anthropic_client") as mock_client:
    mock_client.messages.create = mock.AsyncMock(
        return_value=anthropic_types.Message(
            id="test",
            content=[anthropic_types.TextBlock(text="4", type="text")],
            model="claude-opus-4-1@20250805",
            role="assistant",
            stop_reason="end_turn",
            type="message",
            usage=anthropic_types.Usage(input_tokens=10, output_tokens=2),
        )
    )

    responses = []
    async for response in claude_llm.generate_content_async(
        base_llm_request, stream=False
    ):
      responses.append(response)

    # Verify beta header is sent even when thinking is disabled
    mock_client.messages.create.assert_called_once()
    call_kwargs = mock_client.messages.create.call_args.kwargs
    assert call_kwargs.get("extra_headers") == {
        "anthropic-beta": "interleaved-thinking-2025-05-14"
    }


@pytest.mark.asyncio
async def test_thinking_blocks_preserved_in_assistant_messages(base_llm_request):
  """Test that thinking blocks from previous assistant turn are preserved."""
  from google.adk.models.anthropic_llm import content_to_message_param

  # Create content with thinking block and tool use
  content = types.Content(
      role="model",
      parts=[
          types.Part(
              text="Let me calculate this step by step...", thought=True
          ),
          types.Part.from_function_call(
              name="calculator", args={"expression": "2+2"}
          ),
      ],
  )

  message_param = content_to_message_param(content)

  # Verify message structure
  assert message_param["role"] == "assistant"
  assert len(message_param["content"]) == 2

  # Verify thinking block comes FIRST
  assert message_param["content"][0]["type"] == "thinking"
  assert (
      message_param["content"][0]["thinking"]
      == "Let me calculate this step by step..."
  )

  # Verify tool use comes after
  assert message_param["content"][1]["type"] == "tool_use"
  assert message_param["content"][1]["name"] == "calculator"
