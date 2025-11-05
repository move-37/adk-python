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

"""Tests for Anthropic streaming with thinking support."""

import os
from unittest import mock

from anthropic import types as anthropic_types
from google.adk.models.anthropic_llm import Claude
from google.adk.models.anthropic_llm import streaming_event_to_llm_response
from google.adk.models.llm_request import LlmRequest
from google.genai import types
import pytest


@pytest.fixture
def claude_llm():
  return Claude(model="claude-opus-4-1@20250805", max_tokens=16000)


@pytest.fixture
def llm_request_with_thinking():
  return LlmRequest(
      model="claude-opus-4-1@20250805",
      contents=[
          types.Content(
              role="user",
              parts=[types.Part.from_text(text="Explain quantum computing")],
          )
      ],
      config=types.GenerateContentConfig(
          system_instruction="You are a helpful assistant",
          thinking_config=types.ThinkingConfig(
              include_thoughts=True, thinking_budget=-1
          ),
      ),
  )


def test_streaming_event_text_delta():
  """Test that text_delta events are converted correctly."""

  class TextDelta:
    type = "text_delta"
    text = "Hello "

  class ContentBlockDeltaEvent:
    type = "content_block_delta"
    delta = TextDelta()

  event = ContentBlockDeltaEvent()
  llm_response = streaming_event_to_llm_response(event)

  assert llm_response is not None
  assert llm_response.partial is True
  assert llm_response.content is not None
  assert llm_response.content.role == "model"
  assert len(llm_response.content.parts) == 1
  assert llm_response.content.parts[0].text == "Hello "
  assert (
      not hasattr(llm_response.content.parts[0], "thought")
      or not llm_response.content.parts[0].thought
  )


def test_streaming_event_thinking_delta():
  """Test that thinking_delta events are converted with thought=True."""

  class ThinkingDelta:
    type = "thinking_delta"
    thinking = "Let me think... "

  class ContentBlockDeltaEvent:
    type = "content_block_delta"
    delta = ThinkingDelta()

  event = ContentBlockDeltaEvent()
  llm_response = streaming_event_to_llm_response(event)

  assert llm_response is not None
  assert llm_response.partial is True
  assert llm_response.content is not None
  assert llm_response.content.role == "model"
  assert len(llm_response.content.parts) == 1
  assert llm_response.content.parts[0].text == "Let me think... "
  assert hasattr(llm_response.content.parts[0], "thought")
  assert llm_response.content.parts[0].thought is True


def test_streaming_event_message_delta_usage():
  """Test that message_delta events with usage are converted correctly."""

  class Usage:
    input_tokens = 100
    output_tokens = 50

  class MessageDeltaEvent:
    type = "message_delta"
    usage = Usage()

  event = MessageDeltaEvent()
  llm_response = streaming_event_to_llm_response(event)

  assert llm_response is not None
  assert llm_response.usage_metadata is not None
  assert llm_response.usage_metadata.prompt_token_count == 100
  assert llm_response.usage_metadata.candidates_token_count == 50
  assert llm_response.usage_metadata.total_token_count == 150


def test_streaming_event_ignored():
  """Test that start/stop events are ignored."""

  class MessageStartEvent:
    type = "message_start"

  event = MessageStartEvent()
  llm_response = streaming_event_to_llm_response(event)

  assert llm_response is None
