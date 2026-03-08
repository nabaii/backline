Designing the chat LLM for Backline is really about routing the user’s question to the right analysis module, retrieving the relevant stats, and then letting the LLM explain the findings clearly.

Think of the LLM as the interpreter, not the analyst.

Your modules do the analysis, the LLM explains it.

Core Architecture for Your Chat LLM

The structure should look like this:

User Question
     ↓
Intent Classifier
     ↓
Module Router
     ↓
Analytics Engine (your 5 modules)
     ↓
LLM Explanation Layer
     ↓
User Response

The key is that the LLM never decides the bet.
It summarizes evidence from the modules.

Step 1 — Intent Detection

You first determine which module should handle the question.

Example intents:

User Question	Intent	Module
Is over 2.5 good here?	GOAL_TOTAL	Goal Total Analyzer
Will both teams score?	BTTS	BTTS Analyzer
Can Arsenal win this?	MATCH_WINNER	Match Winner Analyzer
Is this bet safe?	RISK	Risk Analyzer
Over 2.5 or BTTS?	COMPARISON	Market Comparison

Example prompt:

Classify the user's betting question into one of these categories:

GOAL_TOTAL
BTTS
MATCH_WINNER
RISK
COMPARISON

Return only the category.


Step 2 — Module Execution (Implemented)

The chat orchestrator (backend/rag/chat_orchestrator.py) calls the kitchen
workspaces directly — the same code that powers the manual kitchen UI.

Example:

User asks: "Is over 2.5 good for Chelsea vs Villa?"

1. Intent parser returns: {"bet_type": "over_under", "line": 2.5}
2. Orchestrator calls OverUnderWorkspace.get_evidence() for both teams
3. Computes hit rates (over/under counts) from real match data
4. Feeds structured metrics to the LLM explanation layer

Supported workspaces:
- over_under → OverUnderWorkspace (line defaults to 2.5)
- one_x_two → OneXTwoWorkspace (outcome: win/draw/loss)
- double_chance → DoubleChanceWorkspace
- corners → CornerWorkspace (line defaults to 8.5)

Step 3 — Structured Output from Modules

Each module returns structured data, not text.

Example:

{
  "market": "over_2_5",
  "hit_rate_last_10": 0.70,
  "league_average": 0.52,
  ...
}

Step 4 — LLM Explanation Layer

Now the LLM translates the analytics into human language.

Prompt example:

You are a betting research assistant.

Explain the statistics clearly.

Do not recommend bets.
Do not say "best bet".
Focus on trends and probabilities.

DATA:
{retrieved_stats}

USER QUESTION:
{user_question}

Example output:

Over 2.5 goals has landed in 7 of the last 10 matches involving these teams, which is higher than the league average of 52%.

Both sides also rank high in attacking output, with strong expected goal numbers. The main factor supporting higher goal totals is the fast tempo and shot volume these teams generate.

Notice:

No prediction
No recommendation
Just analysis.

Step 5 — Special Handling for “Best Bet”

If the classifier detects:

BEST_BET

Return your philosophy response:

Backline doesn’t select bets.
You steer the bet, and I’ll help analyze the data behind it.

Tell me the market you’re considering.

Step 6 — Comparison Module

If the user asks:

Over 2.5 or BTTS?

You call two analyzers.

Example:

Goal Total Analyzer
BTTS Analyzer

Then feed both outputs into the LLM.

Prompt:

Compare these two betting markets based on the data.
Explain the differences in likelihood and risk.
Do not recommend a bet.
Step 7 — Conversation Memory

The assistant should remember what market the user is discussing.

Example flow:

User:

Over 2.5 for Chelsea vs Villa?

Assistant analyzes.

User:

What about BTTS?

Your system already knows the match.

So only the market changes.

Step 8 — Ideal Prompt Personality

Your system prompt should reinforce Backline philosophy.

Example:

You are Backline, a betting research assistant.

Your role is to analyze football betting markets using statistical evidence.

You DO NOT:
- give predictions
- choose bets
- say "best bet"

You DO:
- explain trends
- highlight risk
- provide context
- help users evaluate their own bets
The Most Important Design Principle

Do not let the LLM access the entire dataset.

Only give it:

Small structured analytics outputs

Example:

Bad:

Send 500 match records

Good:

Send 10 computed metrics

This keeps:

latency low

cost low

hallucinations low

Response Style Guidelines

Every response from Backline must follow these rules:

1. Keep it short and digestible.
   - 2–4 sentences is the ideal length. Never exceed a short paragraph.
   - Lead with the key insight, not a preamble.

2. Teach the user something.
   - Every response should leave the user understanding the data better.
   - Explain *why* the numbers matter, not just what they are.
   - Example: "Chelsea's last 8 home matches averaged 3.1 total goals — driven mainly by their high shot volume (17 per game) rather than clinical finishing."

3. No stat dumps.
   - Do not list every metric from the context. Pick the 2–3 most relevant numbers and weave them into a clear sentence.

4. Use plain language.
   - Avoid jargon unless the user introduced it first.
   - Write like you're explaining to a smart friend, not writing an academic paper.

5. Reference the data, not opinions.
   - Always tie claims back to specific stats from the retrieved matches.
   - Never speculate beyond what the data supports.

6. No filler.
   - No greetings, no "Great question!", no "Let me break this down for you."
   - Start directly with the analysis.

What This Becomes

Your LLM becomes something like:

Betting Copilot

Not:

Betting Predictor

Exactly matching your philosophy:

You steer the boat.
Backline shows the waters.