import re

FILE_PATH = "c:/Users/enaic/OneDrive/Desktop/backline/backline_v2/backend/backend_api.py"

with open(FILE_PATH, "r", encoding="utf-8") as f:
    text = f.read()

# 1. Add to required_features
text = text.replace(
    'required_features=["team_h1_goals", "opponent_h1_goals", "team_h2_goals", "opponent_h2_goals"]',
    'required_features=["team_h1_goals", "opponent_h1_goals", "team_h2_goals", "opponent_h2_goals", "goals_scored", "opponent_goals"]'
)
text = text.replace(
    'required_features=[\n                "team_h1_goals",\n                "opponent_h1_goals",\n            ]',
    'required_features=[\n                "team_h1_goals",\n                "opponent_h1_goals",\n                "goals_scored",\n                "opponent_goals",\n            ]'
)

# 2. Update _build_recent_matches* functions to extract and include goals
# For _build_recent_matches (1x2)
text = re.sub(
    r'(def _build_recent_matches\(.*?for row in df\.to_dict\("records"\):)',
    r'\1\n        goals_scored = float(row.get("goals_scored", 0.0))\n        opponent_goals = float(row.get("opponent_goals", 0.0))',
    text,
    flags=re.DOTALL
)
# The above applies to `_build_recent_matches(` correctly. But wait, `_build_recent_matches` (1x2) doesn't have `goals_scored` yet. Let's do it specifically:
def add_extraction_to_func(func_name, code):
    # Find the function body start
    pattern = rf'(def {func_name}\(.*?\):\n.*?for row in df\.to_dict\("records"\):\n)'
    return re.sub(pattern, r'\g<1>        goals_scored = float(row.get("goals_scored", 0.0))\n        opponent_goals = float(row.get("opponent_goals", 0.0))\n', code, flags=re.DOTALL)

def add_fields_to_dict(func_name, code):
    # Find the dictionary inside rows.append({
    pattern = rf'(def {func_name}\(.*?\):\n(?:.*?)rows\.append\(\s*\{{)(.*?)(?=\*\*\_build_overlay_metrics_from_row)'
    def repl(m):
        inner = m.group(2)
        if '"goals_scored"' not in inner:
            return m.group(1) + inner + '                "goals_scored": goals_scored,\n                "opponent_goals": opponent_goals,\n                '
        return m.group(0)
    return re.sub(pattern, repl, code, flags=re.DOTALL)

func_names = [
    "_build_recent_matches",
    "_build_recent_matches_over_under",
    "_build_recent_matches_team_goals_over_under",
    "_build_recent_matches_corners",
    "_build_recent_matches_double_chance",
    "_build_recent_matches_weh",
    "_build_recent_matches_wbh",
    "_build_recent_matches_first_half_over_under",
    "_build_recent_matches_first_half_one_x_two"
]

for fn in func_names:
    # check if goals_scored is already extracted in the first 5 lines of the loop
    # actually, just insert it if it doesn't exist
    if fn in text:
        text = add_extraction_to_func(fn, text)
        text = add_fields_to_dict(fn, text)

with open(FILE_PATH, "w", encoding="utf-8") as f:
    f.write(text)

print("Done")
