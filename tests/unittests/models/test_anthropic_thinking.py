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
async def test_thinking_budget_automatic(claude_llm, base_llm_request):
  """Test that thinking_budget=-1 uses automatic budget (10000 tokens)."""
  base_llm_request.config.thinking_config = types.ThinkingConfig(
      include_thoughts=True, thinking_budget=-1
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
        base_llm_request, stream=False
    ):
      responses.append(response)

    # Verify thinking parameter was passed with 10000 tokens
    mock_client.messages.stream.assert_called_once()
    call_kwargs = mock_client.messages.stream.call_args.kwargs
    assert call_kwargs["thinking"] == {"type": "enabled", "budget_tokens": 10000}


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
    # Thinking should not be in the call
    assert "thinking" not in mock_client.messages.create.call_args.kwargs


@pytest.mark.asyncio
async def test_thinking_budget_specific_value(claude_llm, base_llm_request):
  """Test that thinking_budget with specific value is used."""
  base_llm_request.config.thinking_config = types.ThinkingConfig(
      include_thoughts=True, thinking_budget=5000
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
        base_llm_request, stream=False
    ):
      responses.append(response)

    # Verify thinking parameter was passed with 5000 tokens
    mock_client.messages.stream.assert_called_once()
    call_kwargs = mock_client.messages.stream.call_args.kwargs
    assert call_kwargs["thinking"] == {"type": "enabled", "budget_tokens": 5000}


@pytest.mark.asyncio
async def test_thinking_budget_minimum_enforced(claude_llm, base_llm_request):
  """Test that thinking budget below 1024 is enforced to minimum 1024."""
  base_llm_request.config.thinking_config = types.ThinkingConfig(
      include_thoughts=True, thinking_budget=500
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
        base_llm_request, stream=False
    ):
      responses.append(response)

    # Verify thinking parameter was enforced to minimum 1024 tokens
    mock_client.messages.stream.assert_called_once()
    call_kwargs = mock_client.messages.stream.call_args.kwargs
    assert call_kwargs["thinking"] == {"type": "enabled", "budget_tokens": 1024}


def test_thinking_block_parsing():
  """Test that thinking blocks are parsed as Part(thought=True)."""

  # Create a mock thinking block
  class ThinkingBlock:

    def __init__(self):
      self.thinking = "Let me think about this step by step..."
      self.type = "thinking"

  thinking_block = ThinkingBlock()
  part = content_block_to_part(thinking_block)

  assert part.text == "Let me think about this step by step..."
  assert hasattr(part, "thought")
  assert part.thought is True


def test_thinking_block_type_check():
  """Test alternative thinking block detection via type attribute."""

  # Create a mock thinking block with only type attribute
  class ThinkingBlockTypeOnly:

    def __init__(self):
      self.type = "thinking"

    def __str__(self):
      return "Thinking content here"

  thinking_block = ThinkingBlockTypeOnly()
  part = content_block_to_part(thinking_block)

  assert part.text == "Thinking content here"
  assert hasattr(part, "thought")
  assert part.thought is True


@pytest.mark.asyncio
async def test_no_thinking_config(claude_llm, base_llm_request):
  """Test that no thinking config results in normal operation."""
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

    # Verify non-streaming mode was used
    mock_client.messages.create.assert_called_once()
