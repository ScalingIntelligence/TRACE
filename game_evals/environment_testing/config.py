"""
Configuration file for Environment Testing Tools

SETUP INSTRUCTIONS:
===================

1. Install the OpenAI library (if not already installed):
   pip install openai

2. Get your OpenAI API key:
   - Go to: https://platform.openai.com/api-keys
   - Create an account if you don't have one
   - Click "Create new secret key"
   - Copy the key (starts with "sk-")

3. Set your API key using ONE of these methods:

   METHOD 1 (RECOMMENDED): Environment variable (secure, session-only)
   --------------------------------------------------------------------
   export OPENAI_API_KEY="sk-..."
   python interactive_test.py
   
   The key will only exist in your current terminal session.
   
   METHOD 2: Interactive prompt
   -----------------------------
   Just run the script - it will ask for your key if not found.
   python interactive_test.py
   
   METHOD 3: Config file (NOT RECOMMENDED on shared clusters)
   -----------------------------------------------------------
   Set OPENAI_API_KEY below (replace "your-api-key-here")
   Only use this if you're on a private machine.

COST INFO:
==========
GPT-4o-mini is very cheap (~$0.15 per 1M input tokens, $0.60 per 1M output tokens)
A typical game costs less than $0.01

You can also use other models by changing OPENAI_MODEL below.
"""

import os

# ============================================================================
# CONFIGURATION
# ============================================================================

# OpenAI API Key for testing against LLM
# Priority: 1) Environment variable, 2) This config value, 3) Interactive prompt
# For security on shared clusters, use environment variable or interactive prompt
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "your-api-key-here")

# Model to use for LLM opponent (default model)
# Options: "gpt-4o-mini" (cheapest), "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo", etc.
# This is used for:
#   - interactive_test.py Mode 3 (Play vs LLM)
#   - interactive_test.py Mode 4 (LLM vs LLM) when both use the same model
OPENAI_MODEL = "gpt-4o-mini"

# ============================================================================
# DUAL MODEL CONFIGURATION (for Model A vs Model B)
# ============================================================================

# Model A configuration (Player 0)
OPENAI_MODEL_A = "gpt-4o-mini"
OPENAI_TEMPERATURE_A = 0.7
OPENAI_MAX_TOKENS_A = 500

# Model B configuration (Player 1)
OPENAI_MODEL_B = "gpt-4o"  # Can use a different/stronger model
OPENAI_TEMPERATURE_B = 0.7
OPENAI_MAX_TOKENS_B = 500

# ============================================================================
# SHARED CONFIGURATION (used when OPENAI_MODEL is used)
# ============================================================================

# Temperature for LLM responses
# 0.0 = deterministic/consistent, 1.0 = creative/varied
# Recommended: 0.7 for game playing
OPENAI_TEMPERATURE = 0.7

# Maximum tokens for LLM responses
# 200 is usually enough for game actions
OPENAI_MAX_TOKENS = 200

