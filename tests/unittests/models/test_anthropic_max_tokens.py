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

"""Tests for Anthropic max_tokens streaming threshold behavior."""

import os
from unittest import mock

from anthropic import types as anthropic_types
from google.adk.models.anthropic_llm import Claude
from google.adk.models.llm_request import LlmRequest
from google.genai import types
import pytest


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
async def test_max_tokens_below_threshold_uses_non_streaming(base_llm_request):
  """Test that max_tokens < 8192 uses non-streaming mode."""
  claude_llm = Claude(model="claude-opus-4-1@20250805", max_tokens=4096)

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
    mock_client.messages.stream.assert_not_called()


@pytest.mark.asyncio
async def test_max_tokens_at_threshold_uses_streaming(base_llm_request):
  """Test that max_tokens >= 8192 uses streaming mode."""
  claude_llm = Claude(model="claude-opus-4-1@20250805", max_tokens=8192)

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
            usage=anthropic_types.Usage(input_tokens=10, output_tokens=2),
        )
    )
    mock_client.messages.stream = mock.Mock(return_value=mock_stream)

    responses = []
    async for response in claude_llm.generate_content_async(
        base_llm_request, stream=False
    ):
      responses.append(response)

    # Verify streaming mode was used
    mock_client.messages.stream.assert_called_once()
    assert not mock_client.messages.create.called


@pytest.mark.asyncio
async def test_max_tokens_above_threshold_uses_streaming(base_llm_request):
  """Test that max_tokens > 8192 uses streaming mode."""
  claude_llm = Claude(model="claude-opus-4-1@20250805", max_tokens=16000)

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
            usage=anthropic_types.Usage(input_tokens=10, output_tokens=2),
        )
    )
    mock_client.messages.stream = mock.Mock(return_value=mock_stream)

    responses = []
    async for response in claude_llm.generate_content_async(
        base_llm_request, stream=False
    ):
      responses.append(response)

    # Verify streaming mode was used
    mock_client.messages.stream.assert_called_once()
    assert not mock_client.messages.create.called


@pytest.mark.asyncio
async def test_stream_flag_overrides_max_tokens(base_llm_request):
  """Test that stream=True forces streaming regardless of max_tokens."""
  claude_llm = Claude(model="claude-opus-4-1@20250805", max_tokens=1024)

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
            usage=anthropic_types.Usage(input_tokens=10, output_tokens=2),
        )
    )
    mock_client.messages.stream = mock.Mock(return_value=mock_stream)

    responses = []
    async for response in claude_llm.generate_content_async(
        base_llm_request, stream=True
    ):
      responses.append(response)

    # Verify streaming mode was used even with low max_tokens
    mock_client.messages.stream.assert_called_once()
    assert not mock_client.messages.create.called


@pytest.mark.asyncio
async def test_thinking_enables_streaming_regardless_max_tokens(
    base_llm_request,
):
  """Test that thinking config enables streaming regardless of max_tokens."""
  claude_llm = Claude(model="claude-opus-4-1@20250805", max_tokens=1024)
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
            usage=anthropic_types.Usage(input_tokens=10, output_tokens=2),
        )
    )
    mock_client.messages.stream = mock.Mock(return_value=mock_stream)

    responses = []
    async for response in claude_llm.generate_content_async(
        base_llm_request, stream=False
    ):
      responses.append(response)

    # Verify streaming mode was used due to thinking config
    mock_client.messages.stream.assert_called_once()
    assert not mock_client.messages.create.called


@pytest.mark.asyncio
async def test_streaming_decision_logic(base_llm_request):
  """Test the complete streaming decision logic."""
  # Case 1: No streaming triggers
  claude_llm_1 = Claude(model="claude-opus-4-1@20250805", max_tokens=4096)

  with mock.patch.object(claude_llm_1, "_anthropic_client") as mock_client:
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
    async for response in claude_llm_1.generate_content_async(
        base_llm_request, stream=False
    ):
      responses.append(response)

    # Should use non-streaming
    mock_client.messages.create.assert_called_once()

  # Case 2: Large max_tokens trigger
  claude_llm_2 = Claude(model="claude-opus-4-1@20250805", max_tokens=10000)

  with mock.patch.object(claude_llm_2, "_anthropic_client") as mock_client:
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
            usage=anthropic_types.Usage(input_tokens=10, output_tokens=2),
        )
    )
    mock_client.messages.stream = mock.Mock(return_value=mock_stream)

    responses = []
    async for response in claude_llm_2.generate_content_async(
        base_llm_request, stream=False
    ):
      responses.append(response)

    # Should use streaming
    mock_client.messages.stream.assert_called_once()
