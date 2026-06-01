# Notebook Authoring Spec for Agents

## Status
- document_type: notebook_authoring_spec
- scope: `slides/` subtree
- target_files:
  - `slides/**/*.ipynb`
  - `slides/**/*.py` (Jupytext percent format)
- primary_audience: AI agents
- secondary_audience: human instructors
- source_of_truth: `.py` Jupytext file
- derived_artifact: `.ipynb`
- python_env: `slides/.venv`

## Core Rule
在 `slides/` 下编写课程 notebook 时，**优先维护 Jupytext percent-format 的 `.py` 文件**，再由该 `.py` 文件同步生成 `.ipynb`。

Do:
- edit: source `.py` files (e.g., `topic_name.py`)
- generate/sync: derived `.ipynb` files (e.g., `topic_name.ipynb`)

Do not:
- 只改 `.ipynb` 而不回写 `.py`
- 把 `.ipynb` 当作唯一真实来源

## Rationale
使用 `.py` 作为 source of truth 的原因：
- 更适合 Git diff；
- 更适合 AI 生成和重写；
- 更适合批量重构 notebook 结构；
- 更容易显式维护 cell tags 和教学结构。

## Required Tooling
Assume the `slides/` project uses:
- `uv`
- `jupyter`
- `jupytext`
- `ipykernel`
- `nbformat`

## Directory Convention
For each course topic or module, use a dedicated folder:

```text
slides/
├── pyproject.toml
├── .venv/
├── ipynb_agent_spec.md
└── topic_name/
    ├── figures/
    ├── notebook_name.py
    └── notebook_name.ipynb
```

## File Naming Convention
*Currently, there are no strict file naming restrictions.*

Please name your folders and files clearly according to the semantic topic or pedagogical purpose (e.g., `decision_trees/01_intro.py`, `linear_regression/main_class.py`, or `lesson01/lab.py`).

If there are multiple notebooks in the same topic, use explicit suffixes to distinguish their roles:
- `topic_main.py`
- `topic_demo.py`
- `topic_lab.py`

## Required Notebook Format
The `.py` source file should use Jupytext percent format.

Expected header pattern:

```python
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---
```

## Teaching Model
Each notebook is a classroom artifact, not just a code scratchpad.

Notebook responsibilities:
1. carry the classroom flow;
2. trigger questions before explanations;
3. show phenomena through plots/tables/code execution;
4. preserve teacher-only script notes when needed;
5. keep text short and stage-like.

## Cell Taxonomy
Every major notebook should use these 5 core cell roles.

### 1. `meta`
Purpose:
- notebook title
- week number
- learning goals
- usage note

Typical format:
- markdown cell

### 2. `prompt`
Purpose:
- ask a question before running code (predict/create tension)
- or force a judgment/pause after an output (discuss/reflect)
- drives all student interaction

Typical format:
- markdown cell

### 3. `stage`
Purpose:
- produce the main classroom event
- code + figure/table/output

Typical format:
- code cell

### 4. `takeaway`
Purpose:
- give a short interpretation of the output (name the phenomenon)
- elevate from "what we just saw" to "what this means"
- transform a phenomenon into a transferable judgment
- give students a sentence they can reuse for decision-making

Typical format:
- markdown cell

### 5. `script`
Purpose:
- teacher-only speaking notes
- reminders about pacing, likely confusion points, or follow-up prompts

Typical format:
- markdown cell
- should include tag `teacher-only`
- should preferably be hidden/collapsible in the frontend

## Required Tags
Use tags consistently.

Minimum tag set:
- `meta`
- `prompt`
- `stage`
- `takeaway`
- `script`
- `teacher-only`

Examples:
- pre/post interaction cell: `tags=["prompt"]`
- main demo cell: `tags=["stage"]`
- phenomenon & judgment cell: `tags=["takeaway"]`
- teacher note cell: `tags=["script", "teacher-only"]`

## Structure Rule
The default teaching rhythm inside each scene should follow this pattern:

```text
prompt → stage → takeaway
```

Across scenes, the notebook should be woven by standard markdown narrative text (no special tags needed) that explicitly:
- restates the core thread;
- connects the dots;
- prepares the semantic shift to the next section.

## Content Density Rule
- 每个 markdown cell 只承载一个主要功能；
- 每个 prompt cell 最多 1~3 个问题；
- 每个 takeaway cell 必须兼顾“解释现象”和“业务升华”，但尽量精简；
- 每个 stage cell 尽量只产生一个主要图或一个主要表；
- 如果代码太长，应拆分为“setup cell + stage cell”。

## Prompt Writing Rule
Prompts should not be "terminology questions" that only make sense after you already understand the concept.

A good prompt must provide enough situational context that a student can form a prediction or reflection.

Patterns to use:
- **Business scenario**: "假设你是一家超市的数据负责人，老板给你三个模型..."
- **Concrete analogy**: "同一个考试大纲，但每次给你不同的练习题，好学生应该每次考出接近的分数..."
- **Economic stakes**: "如果今天必须选一个上线，选错了会怎样？"
- **Visual prediction**: "不看后面的图，你觉得训练误差会怎么走？"

Avoid:
- Prompts that are answerable only by students who already know the concept ("> 谁像欠拟合？谁像过拟合？" — too flat)
- Prompts that are too broad or open-ended without a concrete anchor

## Takeaway Writing Rule
A `takeaway` cell must do two things at once:

1. **Name the phenomenon**: "This is why we call it high variance. The left panel is tighter because..."
2. **Elevate to judgment**: "So what does this mean for model selection? In any problem where training samples are limited..."

It should connect to risk, decision-making, or real-world stakes.
A scene that only names the phenomenon feels unfinished.
A scene that includes a transferable judgment makes students feel they learned something they can use.

## Narrative Arc Design Rule
A classroom notebook is not a flat list of demos.
It is a deliberately designed cognitive journey.

A 90-minute session should have roughly:
- 5–7 scenes;
- 1–2 synthesis sections;
- a clear prologue that sets stakes;
- a clear synthesis before transitioning to the next topic.

The high-level narrative arc should follow:

```text
prologue (stakes) → scenes (prompt → stage → takeaway) → synthesis (anchor) → scenes → synthesis (final) → next-topic transition
```

"More scenes" does not mean "more code."
It means more distinct cognitive beats—each scene should produce one main observation that feeds into one takeaway.

## Plotting Rule
For classroom plots:
- prefer one strong figure over many small ones;
- set titles and axis labels explicitly;
- annotate the interpretation if needed;
- avoid tiny unreadable defaults;
- prefer deterministic random seeds.

## Data / Reproducibility Rule
- always set random seed in demo notebooks;
- keep datasets lightweight unless explicitly needed;
- do not require secret credentials;
- do not depend on network downloads during class unless explicitly planned.

## Output Rule
Default policy:
- commit the `.ipynb` file for convenience;
- do not rely on embedded outputs as the only source of truth;
- prefer notebooks that can be re-executed from top to bottom.

If not explicitly requested:
- avoid embedding excessive binary output;
- avoid extremely large notebook outputs.

## Teacher-Only Script Rule
Teacher notes should:
- be short;
- focus on pacing or prompting;
- not duplicate the visible explanation cells;
- be tagged with `teacher-only`.

Example content:
- "先让学生猜 20 秒，不要立刻运行。"
- "如果学生答不出 variance，就先说‘模型不稳定’。"
- "这里强调：训练误差下降不代表泛化能力提高。"

## Preferred Lesson/Module Notebook Skeleton
Recommended high-level structure:

```text
00_meta
00_prologue (stakes, business scenario, why this module exists)
01_scene_1 (prompt → stage → takeaway)
02_scene_2
03_scene_3
04_synthesis_1 (cognitive anchor: connect first half of class)
05_scene_4
06_scene_5
07_scene_6
08_synthesis_2 (final synthesis: collect all judgments)
09_transition (natural question → next topic)
```

The exact number of scenes can vary by module content, but the arc should always include:
- a prologue;
- at least one synthesis section;
- a final synthesis that collects all takeaways;
- a transition that makes the next topic feel like a natural answer to an open question.

## Commands
### Create/update notebook from source
From repository root:

```bash
uv run --directory slides jupytext --sync topic_dir/notebook_name.py
```

### Convert source file into notebook explicitly
```bash
uv run --directory slides jupytext --to ipynb topic_dir/notebook_name.py -o topic_dir/notebook_name.ipynb
```

### Execute notebook for validation
```bash
uv run --directory slides jupyter nbconvert --to notebook --execute --inplace topic_dir/notebook_name.ipynb
```

## Agent Checklist
Before finishing notebook generation, an agent should verify:
1. Is the `.py` file present and readable as the source of truth?
2. Does the notebook use the 5 core cell tags?
3. Is the narrative structured around `prompt → stage → takeaway`?
4. Does every major scene include a `takeaway` cell that both names the phenomenon and elevates to judgment?
5. Are narrative syntheses placed at key transitions, not skipped?
6. Are prompts written with situational context, not just terminology?
7. Are teacher-only notes separated from student-visible text?
8. Does the notebook run top-to-bottom without external hidden state?
9. Has the `.ipynb` been synced from the `.py` source?
10. Are plots/tables legible enough for classroom projection?
11. Does the notebook end with a transition question that makes the next topic feel like a natural continuation?
