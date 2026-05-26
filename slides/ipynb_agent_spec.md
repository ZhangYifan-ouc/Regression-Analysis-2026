# Notebook Authoring Spec for Agents

## Status
- document_type: notebook_authoring_spec
- scope: `slides/` subtree
- target_files:
  - `slides/weekXX_new/*.ipynb`
  - `slides/weekXX_new/*.py` (Jupytext percent format)
- primary_audience: AI agents
- secondary_audience: human instructors
- source_of_truth: `.py` Jupytext file
- derived_artifact: `.ipynb`
- python_env: `slides/.venv`

## Core Rule
在 `slides/` 下编写课程 notebook 时，**优先维护 Jupytext percent-format 的 `.py` 文件**，再由该 `.py` 文件同步生成 `.ipynb`。

Do:
- edit: `weekXX_class.py`
- generate/sync: `weekXX_class.ipynb`

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
For each week notebook package, use:

```text
slides/
├── pyproject.toml
├── .venv/
├── ipynb_agent_spec.md
└── weekXX_new/
    ├── figures/
    ├── weekXX_class.py
    └── weekXX_class.ipynb
```

## File Naming Convention
Preferred names:
- source file: `week12_class.py`
- notebook file: `week12_class.ipynb`

If there are multiple notebooks in the same week, use explicit suffixes:
- `week12_class_main.py`
- `week12_class_demo.py`
- `week12_lab.py`

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
Every major notebook should use these cell roles.

### 1. `meta`
Purpose:
- notebook title
- week number
- learning goals
- usage note

Typical format:
- markdown cell

### 2. `cue`
Purpose:
- ask a question before running code
- prompt prediction
- create tension

Typical format:
- markdown cell

### 3. `stage`
Purpose:
- produce the main classroom event
- code + figure/table/output

Typical format:
- code cell

### 4. `explain`
Purpose:
- give a short interpretation after the output appears
- connect phenomenon to concept

Typical format:
- markdown cell

### 5. `takeaway`
Purpose:
- elevate from "what we just saw" to "what this means"
- transform a phenomenon into a transferable judgment
- give students a sentence they can reuse for decision-making

This cell is the single most important cognitive leap in each scene.
Without it, the notebook stops at naming and never reaches judgment.

Typical format:
- markdown cell

Key distinction: `explain` names the phenomenon; `takeaway` tells you why it matters.

### 6. `checkpoint`
Purpose:
- force students to make a judgment
- insert a pause for discussion or mini reflection

Typical format:
- markdown cell

### 7. `interlude`
Purpose:
- recap and synthesize across multiple preceding scenes
- create a deliberate cognitive anchor before the next major section
- help students see the connections between scenes they just experienced

An interlude should not introduce new code or new experiments.
It should collect, compress, and reframe.

Typical format:
- markdown cell

### 8. `script`
Purpose:
- teacher-only speaking notes
- reminders about pacing, likely confusion points, or follow-up prompts

Typical format:
- markdown cell
- should include tag `teacher-only`
- should preferably be hidden/collapsible in the frontend

### 9. `backup`
Purpose:
- optional extensions
- extra experiments
- alternative explanation branches

Typical format:
- markdown or code cell

## Required Tags
Use tags consistently.

Minimum tag set:
- `meta`
- `cue`
- `stage`
- `explain`
- `takeaway`
- `checkpoint`
- `interlude`
- `script`
- `teacher-only`
- `backup`

Examples:
- markdown prompt cell: `tags=["cue"]`
- phenomenon-to-judgment cell: `tags=["takeaway"]`
- scene synthesis cell: `tags=["interlude"]`
- teacher note cell: `tags=["script", "teacher-only"]`
- main demo cell: `tags=["stage"]`
- optional appendix cell: `tags=["backup"]`

## Structure Rule
The default teaching rhythm inside each scene should follow this pattern:

```text
cue → stage → explain → takeaway → (checkpoint)
```

Not every scene must include all five, but `takeaway` should rarely be omitted.

Across scenes, the notebook should be woven by interludes that explicitly:
- restate the core thread;
- connect the dots;
- prepare the semantic shift to the next section.

## Content Density Rule
- 每个 markdown cell 只承载一个主要功能；
- 每个 cue cell 最多 1~3 个问题；
- 每个 explain cell 尽量控制在 3~6 行；
- 每个 stage cell 尽量只产生一个主要图或一个主要表；
- 如果代码太长，应拆分为“setup cell + stage cell”。

## Cue Writing Rule
Cues should not be "terminology questions" that only make sense after you already understand the concept.

A good cue must provide enough situational context that a student can form a prediction before seeing code.

Patterns to use:
- **Business scenario**: "假设你是一家超市的数据负责人，老板给你三个模型..."
- **Concrete analogy**: "同一个考试大纲，但每次给你不同的练习题，好学生应该每次考出接近的分数..."
- **Economic stakes**: "如果今天必须选一个上线，选错了会怎样？"
- **Visual prediction**: "不看后面的图，你觉得训练误差会怎么走？"

Avoid:
- Cues that are answerable only by students who already know the concept ("> 谁像欠拟合？谁像过拟合？" — too flat)
- Cues that are too broad or open-ended without a concrete anchor

## Explain vs Takeaway Rule
These two cell types must NOT be confused.

### `explain` responsibilities:
- Name the phenomenon: "This is why we call it high variance."
- Connect to the immediate output: "The left panel is tighter because..."
- Be short (3–6 lines).

### `takeaway` responsibilities:
- Elevate to judgment: "So what does this mean for model selection?"
- Make it transferable: "In any problem where training samples are limited and noisy..."
- Connect to risk, decision-making, or real-world stakes.
- Often starts with: "This means..."、"The real danger here is..."、"So the choice is not about..."

A scene that ends only at `explain` feels unfinished.
A scene that includes `takeaway` makes students feel they learned something they can use.

## Narrative Arc Design Rule
A classroom notebook is not a flat list of demos.
It is a deliberately designed cognitive journey.

A 90-minute session should have roughly:
- 5–7 scenes;
- 1–2 interludes;
- a clear prologue that sets stakes;
- a clear synthesis before transitioning to the next week.

The high-level narrative arc should follow:

```text
prologue (stakes) → scenes (tension → reveal → judgment) → interlude (anchor) → scenes → interlude (final synthesis) → next-week transition
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

## Preferred Week Notebook Skeleton
Recommended high-level structure:

```text
00_meta
00_prologue (stakes, business scenario, why this week exists)
01_scene_1 (tension → reveal → takeaway)
02_scene_2
03_scene_3
04_interlude_1 (cognitive anchor: connect first half of class)
05_scene_4
06_scene_5
07_scene_6
08_interlude_2 (final synthesis: collect all judgments)
09_transition (natural question → next week)
99_backup
```

The exact number of scenes can vary by week content, but the arc should always include:
- a prologue;
- at least one interlude;
- a final synthesis that collects all takeaways;
- a transition that makes the next week's topic feel like a natural answer to an open question.

## Commands
### Create/update notebook from source
From repository root:

```bash
uv run --directory slides jupytext --sync week12_new/week12_class.py
```

### Convert source file into notebook explicitly
```bash
uv run --directory slides jupytext --to ipynb week12_new/week12_class.py -o week12_new/week12_class.ipynb
```

### Execute notebook for validation
```bash
uv run --directory slides jupyter nbconvert --to notebook --execute --inplace week12_new/week12_class.ipynb
```

## Agent Checklist
Before finishing notebook generation, an agent should verify:
1. Is the `.py` file present and readable as the source of truth?
2. Does the notebook use the required cell tags?
3. Is the narrative structured around `cue → stage → explain → takeaway → (checkpoint)`?
4. Does every major scene include a `takeaway` cell that elevates from phenomenon to judgment?
5. Are interludes placed at key transitions, not skipped?
6. Are cues written with situational context, not just terminology?
7. Are teacher-only notes separated from student-visible explanation?
8. Does the notebook run top-to-bottom without external hidden state?
9. Has the `.ipynb` been synced from the `.py` source?
10. Are plots/tables legible enough for classroom projection?
11. Does the notebook end with a transition question that makes the next week feel like a natural continuation?
