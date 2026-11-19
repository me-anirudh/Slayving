# 📊 Data Analyst & Analytics Engineer Interview Question Bank
**Target System:** Kuku Promo Finder — Phase 1 (`ast_builder`) Engine  
**Target Roles:** Data Analyst, Analytics Engineer, Product Data Analyst, Data Science Associate  
**Focus:** Python Data Structures, Data Ingestion/Cleaning, SQL & Relational Modeling, Data Quality/Validation, Exploratory Data Analysis, Statistical Rigor, Metrics & A/B Testing.

---

## Section 1: Overview & Competency Framework

For a Data Analyst or Analytics Engineer, the `ast_builder` is evaluated as an **unstructured text ETL pipeline** that ingests messy, heterogeneous script data and transforms it into a clean, canonical analytical asset (`01_structured_episodes.json`).

Interviewers will test your ability to:
1. **Model and Query Semi-Structured Data:** Flatten nested JSON into relational tables, write complex analytical SQL queries with window functions, and design schemas.
2. **Ensure Data Quality & Integrity:** Detect silent data failures, prevent loss of precision during time conversions, and enforce business invariants.
3. **Conduct Rigorous Exploratory Data Analysis (EDA):** Analyze screenplay distributions (dialogue-to-action ratios, scene pacing, speaker turns) and identify anomalies.
4. **Design Experiments & Metric Frameworks:** Measure parser classification accuracy, design A/B tests for downstream editor efficiency, and evaluate classification confusion matrices.

---

## Section 2: Complete Question Bank with Metadata & Defenses

### Domain 1: Data Ingestion, Cleaning & Transformation (Python & Pandas)

#### Q-DA-01 [P0 | Intermediate | Implementation | Project-Derived Prediction]
**Question:** In `parser.py`, you convert screenplay timestamps (`MM:SS.ss`) into integer centiseconds (`parse_timestamp_to_cs`) rather than floating-point seconds. Why is floating-point arithmetic problematic for analytical pipelines, and how does integer centisecond representation solve it?
* **Why This Priority (P0):** Fundamental data engineering decision underpinning the entire pipeline's mechanical determinism.
* **Evidence:** Common industry interview question (Pocket FM, InMobi) & project-derived.
* **Technical Defense / Expected Answer:**
  - *IEEE 754 Floating-Point Limitation:* In Python, standard floats are IEEE 754 64-bit binary floating-point numbers. Many decimal fractions cannot be represented exactly in binary (e.g., `0.1 + 0.2 = 0.30000000000000004`).
  - *Pipeline Drift:* If downstream duration filters execute comparisons like `start_time + duration == end_time` or `total_duration <= 60.0`, floating-point rounding errors accumulate across thousands of screenplay elements, causing boundary checks to fail spuriously.
  - *Integer Centisecond Solution:* Microdrama scripts specify time to two decimal places ($0.01\text{s}$). Multiplying by 100 transforms the problem domain entirely into the set of integers ($\mathbb{Z}$):
    $$\text{cs} = (h \times 360,000) + (m \times 6,000) + (s \times 100) + \text{centis}$$
    Integer addition, subtraction, and comparison in Python are exact and suffer zero precision loss or drift.

#### Q-DA-02 [P0 | Intermediate | Coding | Common Industry Question]
**Question:** Given a loaded `01_structured_episodes.json` dataset, write a Python / Pandas snippet to flatten the nested JSON into a single analytical DataFrame where each row represents one screenplay element, containing episode number, scene number, speaker, element type, and duration in seconds.
* **Why This Priority (P0):** Directly tests the ability to manipulate semi-structured JSON into a tabular structure for analytical consumption.
* **Evidence:** Reported in Data Analyst rounds at Swiggy, Meesho, and Kuku FM.
* **Technical Defense / Expected Code:**
  ```python
  import json
  import pandas as pd

  def flatten_ast_to_dataframe(json_path: str) -> pd.DataFrame:
      with open(json_path, "r", encoding="utf-8") as f:
          data = json.load(f)
      
      rows = []
      for ep in data.get("episodes", []):
          ep_num = ep["episode_num"]
          for sc in ep.get("scenes", []):
              sc_num = sc["scene_num"]
              sc_heading = sc["heading"]
              for el in sc.get("elements", []):
                  dlg_meta = el.get("dialogue_meta") or {}
                  rows.append({
                      "element_id": el["element_id"],
                      "episode_num": ep_num,
                      "scene_num": sc_num,
                      "scene_heading": sc_heading,
                      "element_type": el["element_type"],
                      "speaker": dlg_meta.get("speaker", "N/A") if el["element_type"] == "DIALOGUE" else None,
                      "delivery": dlg_meta.get("delivery"),
                      "start_cs": el["start_cs"],
                      "end_cs": el["end_cs"],
                      "duration_seconds": el["duration_seconds"],
                      "content": el["content"],
                      "word_count": len(el["content"].split())
                  })
      return pd.DataFrame(rows)
  ```

#### Q-DA-03 [P1 | Advanced | Debugging | Project-Derived Prediction]
**Question:** In `UniversalScreenplayParser._parse_microdrama_screenplay`, what happens if a script contains overlapping elements (e.g., an `ACTION` cue running from `00:01.00` to `00:04.80` while a `CAMERA` cue runs from `00:00.00` to `00:04.80`)? How would you identify, quantify, and analyze concurrency in the dataset?
* **Why This Priority (P1):** Directly exposes the difference between serialized text lines and multi-track audio-visual concurrency.
* **Evidence:** Project-derived insight from `ep01.txt` (lines 14–20).
* **Technical Defense / Expected Answer:**
  - *Nature of the Data:* In video production, camera movement, background audio, and physical action happen *simultaneously*. The screenplay reflects this: `EP01_EL001` (Camera) and `EP01_EL002` (Action) overlap in time.
  - *Analytical Trap:* Simply summing `duration_seconds` across all elements overestimates total episode runtime ($114\text{s}$ of content in an $80\text{s}$ scene).
  - *Quantifying Concurrency with Pandas:*
    ```python
    # For any given time slice t (in centiseconds), calculate active concurrent tracks
    # An element is active if start_cs <= t < end_cs
    def compute_concurrency_profile(df_elements: pd.DataFrame, episode: int):
        ep_df = df_elements[df_elements["episode_num"] == episode]
        max_time = ep_df["end_cs"].max()
        # Create 1-second interval timeline
        time_bins = list(range(0, max_time + 100, 100))
        concurrency = []
        for t in time_bins:
            active_tracks = ep_df[(ep_df["start_cs"] <= t) & (ep_df["end_cs"] > t)]
            concurrency.append({
                "time_cs": t,
                "active_elements_count": len(active_tracks),
                "active_types": list(active_tracks["element_type"].unique())
            })
        return pd.DataFrame(concurrency)
    ```

---

### Domain 2: Data Quality, Validation & Silent Failure Detection

#### Q-DA-04 [P0 | Advanced | Scenario-Based | Reported Industry Question]
**Question:** What is a "silent data quality failure" in the context of the `ast_builder`, and how does the `ValidationAudit` dataclass in your code prevent it?
* **Why This Priority (P0):** Central theme in analytics engineering rounds (Zomato, Swiggy, Pocket FM).
* **Evidence:** Reported by Data Platform teams across high-growth startups.
* **Technical Defense / Expected Answer:**
  - *Definition of Silent Failure:* A pipeline failure where the code executes successfully with return code 0, but the output data is corrupt, truncated, or structurally compromised without throwing an exception.
  - *Concrete Screenplay Example:* A regex fails to match a dialogue line because of an unescaped curly apostrophe or irregular markdown header (`###SCENE1` without space). The parser skips the line and continues. An entire dramatic climax is dropped silently.
  - *The Architectural Defense in `parser.py`:*
    1. `ValidationAudit` checks: Counts scenes, cameras, actions, sounds, and dialogues per episode.
    2. Monotonicity Assertion: Checks $sc.\text{start\_cs} < sc.\text{end\_cs}$ and $el.\text{start\_cs} < el.\text{end\_cs}$.
    3. Scene Containment Check: Verifies that every element falls within its parent scene window ($el.\text{end\_cs} \le sc.\text{end\_cs} + 100$).
    4. Hard Fail Mechanism: In `build_and_export_ast`, if $\sum \text{timestamp\_errors} > 0$, the parser raises a hard `ValueError` instead of silently emitting bad JSON.

#### Q-DA-05 [P1 | Intermediate | Statistical / Quality | Project-Derived Prediction]
**Question:** Suppose an upstream authoring team updates their script formatting tool, and in Episode 34, 40% of dialogue lines are miscategorized as `ACTION`. As a Data Analyst, how would you design an automated anomaly detection check to flag this batch before downstream ingestion?
* **Why This Priority (P1):** Tests data distribution monitoring and statistical process control.
* **Evidence:** Common industry practice (Great Expectations, Monte Carlo, Soda SQL).
* **Technical Defense / Expected Answer:**
  - *Baseline Distribution Profiling:* From the historical 50-episode baseline, compute the element distribution ratios:
    - Camera: $\sim 47.1\%$ ($1,754 / 3,727$)
    - Dialogue: $\sim 39.9\%$ ($1,487 / 3,727$)
    - Action: $\sim 10.0\%$ ($372 / 3,727$)
    - Sound FX: $\sim 3.0\%$ ($114 / 3,727$)
  - *Statistical Z-Score / IQR Bounds:* For each new episode, calculate the Dialogue Ratio $R_{\text{dlg}} = \frac{N_{\text{dialogue}}}{N_{\text{total}}}$.
    Using historical mean $\mu = 0.399$ and standard deviation $\sigma \approx 0.045$, if an incoming episode has $Z = \frac{R_{\text{dlg}} - \mu}{\sigma} < -3.0$, an automated alert fires.
  - *Chi-Squared ($\chi^2$) Goodness-of-Fit Test:* Compare observed categorical counts $[O_{\text{cam}}, O_{\text{act}}, O_{\text{sfx}}, O_{\text{dlg}}]$ against expected frequencies $[E_i]$. If $p < 0.01$, flag the file for manual QA review.

---

### Domain 3: SQL & Relational Data Modeling

#### Q-DA-06 [P0 | Intermediate | Architecture & SQL | Reported Industry Question]
**Question:** How would you design a normalized relational schema (PostgreSQL) to store the data generated by `ast_builder`? Write the SQL DDL for `episodes`, `scenes`, and `screenplay_elements`.
* **Why This Priority (P0):** Core competency for Data Analysts and Analytics Engineers; translates JSON AST into enterprise data warehouse modeling.
* **Evidence:** Reported in Pocket FM, Kuku FM, and Zepto technical assessments.
* **Technical Defense / Expected DDL:**
  ```sql
  -- 1. Dimension Table: Episodes
  CREATE TABLE dim_episodes (
      episode_num INT PRIMARY KEY,
      title VARCHAR(255) NOT NULL,
      logline TEXT,
      continuity_notes TEXT,
      total_duration_cs INT NOT NULL,
      created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
  );

  -- 2. Dimension Table: Scenes
  CREATE TABLE dim_scenes (
      scene_id VARCHAR(50) PRIMARY KEY, -- e.g. 'EP01_SC01'
      episode_num INT REFERENCES dim_episodes(episode_num) ON DELETE CASCADE,
      scene_num INT NOT NULL,
      heading VARCHAR(255) NOT NULL,
      start_cs INT NOT NULL,
      end_cs INT NOT NULL,
      duration_cs INT NOT NULL,
      CONSTRAINT chk_scene_monotonic CHECK (end_cs > start_cs)
  );

  -- 3. Fact Table: Screenplay Elements
  CREATE TYPE element_type_enum AS ENUM ('CAMERA', 'ACTION', 'SOUND FX', 'DIALOGUE');

  CREATE TABLE fact_screenplay_elements (
      element_id VARCHAR(50) PRIMARY KEY, -- e.g. 'EP01_EL010'
      episode_num INT REFERENCES dim_episodes(episode_num),
      scene_id VARCHAR(50) REFERENCES dim_scenes(scene_id),
      element_type element_type_enum NOT NULL,
      speaker VARCHAR(100),               -- NULL for camera/action
      modifier VARCHAR(50),              -- e.g. 'O.S.', 'V.O.'
      delivery VARCHAR(100),             -- e.g. 'Cold, Commanding'
      start_cs INT NOT NULL,
      end_cs INT NOT NULL,
      duration_cs INT NOT NULL,
      duration_seconds NUMERIC(6, 2) NOT NULL,
      content TEXT NOT NULL,
      word_count INT NOT NULL,
      raw_tag VARCHAR(100),
      CONSTRAINT chk_element_monotonic CHECK (end_cs > start_cs)
  );

  CREATE INDEX idx_elements_episode_type ON fact_screenplay_elements (episode_num, element_type);
  CREATE INDEX idx_elements_speaker ON fact_screenplay_elements (speaker) WHERE speaker IS NOT NULL;
  ```

#### Q-DA-07 [P0 | Advanced | Coding (SQL) | Reported Industry Question]
**Question:** Write a SQL query using window functions over `fact_screenplay_elements` to:
1. Find the top 3 characters with the highest total dialogue duration in Episode 1.
2. Calculate the average pause (in seconds) before each character speaks (time between previous dialogue's `end_cs` and current dialogue's `start_cs`).
* **Why This Priority (P0):** Tests advanced SQL windowing (`SUM() OVER()`, `LAG() OVER()`) and analytical aggregation.
* **Evidence:** Standard SQL round prompt at media streaming and tech startups.
* **Technical Defense / Expected SQL:**
  ```sql
  WITH dialogue_stream AS (
      SELECT 
          element_id,
          speaker,
          start_cs,
          end_cs,
          duration_cs,
          -- Lag to get the end of the previous dialogue in the same episode
          LAG(end_cs) OVER (
              PARTITION BY episode_num 
              ORDER BY start_cs ASC
          ) AS prev_dialogue_end_cs
      FROM fact_screenplay_elements
      WHERE episode_num = 1 
        AND element_type = 'DIALOGUE'
  ),
  character_metrics AS (
      SELECT 
          speaker,
          COUNT(*) AS total_lines,
          SUM(duration_cs) / 100.0 AS total_speaking_seconds,
          -- Average pause before speaking (ignoring the very first line where lag is null)
          AVG(
              CASE 
                  WHEN prev_dialogue_end_cs IS NOT NULL AND start_cs >= prev_dialogue_end_cs 
                  THEN (start_cs - prev_dialogue_end_cs) / 100.0 
                  ELSE NULL 
              END
          ) AS avg_pause_before_speech_seconds
      FROM dialogue_stream
      GROUP BY speaker
  )
  SELECT 
      speaker,
      total_lines,
      total_speaking_seconds,
      ROUND(avg_pause_before_speech_seconds, 2) AS avg_pause_seconds,
      DENSE_RANK() OVER (ORDER BY total_speaking_seconds DESC) AS rank
  FROM character_metrics
  ORDER BY total_speaking_seconds DESC
  LIMIT 3;
  ```

---

### Domain 4: Exploratory Data Analysis & Pacing Metrics

#### Q-DA-08 [P1 | Intermediate | Analytical Reasoning | Project-Derived Prediction]
**Question:** In microdrama advertising, a high-converting scene is often characterized by rapid dialogue pacing (quick back-and-forth turns). How would you define a "Pacing Index" mathematically using the AST data, and how would you identify the fastest-paced scenes across all 50 episodes?
* **Why This Priority (P1):** Demonstrates ability to translate business/editorial concepts into quantifiable analytical metrics.
* **Evidence:** Direct match for content optimization analytics at Kuku FM and Pocket FM.
* **Technical Defense / Expected Answer:**
  - *Mathematical Metric Definition:*
    $$\text{Pacing Index (PI)} = \frac{\text{Dialogue Turns}}{\text{Scene Duration (Minutes)}} \times \left(1 + \frac{\text{Action Interruptions}}{\text{Dialogue Turns}}\right)$$
    Alternatively, Words Per Second (WPS) within dialogue blocks:
    $$\text{WPS} = \frac{\sum \text{Words}}{\sum \text{Duration (Seconds)}}$$
  - *Analytical Implementation:*
    A scene with 12 rapid one-liner retorts in 20 seconds has $\text{PI} = \frac{12}{0.33} = 36$ turns/min, indicating high dramatic friction. Conversely, a monolithic monologue of 60 seconds with 1 speaker turn has $\text{PI} = 1.0$, indicating low promotional hook potential.
  - *Pandas Pipeline:*
    ```python
    def compute_scene_pacing(df):
        dialogues = df[df["element_type"] == "DIALOGUE"]
        scene_summary = dialogues.groupby(["episode_num", "scene_num"]).agg(
            dialogue_turns=("element_id", "count"),
            unique_speakers=("speaker", "nunique"),
            total_duration_sec=("duration_seconds", "sum"),
            total_words=("word_count", "sum")
        ).reset_index()
        
        scene_summary["turns_per_minute"] = (
            scene_summary["dialogue_turns"] / (scene_summary["total_duration_sec"] / 60.0)
        )
        return scene_summary.sort_values(by="turns_per_minute", ascending=False)
    ```

---

### Domain 5: Evaluation, Metrics & Experimentation (A/B Testing)

#### Q-DA-09 [P0 | Advanced | Experimentation | Reported Industry Question]
**Question:** Suppose you want to test whether switching from your regex-based parser (`Parser A`) to an LLM-based parser (`Parser B`) improves video ad performance. How would you design the A/B test? What are your Primary, Secondary, and Guardrail metrics?
* **Why This Priority (P0):** Tests end-to-end business acumen, experimental design, and metric hierarchy.
* **Evidence:** Reported product data analyst interview prompt at Kuku FM, ShareChat, and Swiggy.
* **Technical Defense / Expected Answer:**
  - *Unit of Randomization:* Ad Creative Level (or Audience Split randomly assigned via Ad Network split-testing). Randomizing at the user impression level prevents creative spillover.
  - *Primary Metric:* **3-Second Hook Retention Rate** (Percentage of impressions watching past second 3.0) and **Cost Per Install (CPI)**.
  - *Secondary Metrics:* Click-Through Rate (CTR) to App Store, Video Completion Rate (VCR), 24-Hour In-App Episode Unlock Rate.
  - *Guardrail Metrics:*
    1. **Parsing Cost:** Total API compute cost per episode ($<\$0.05$ target).
    2. **Format Validity:** 100% compliance with $20\text{s}–60\text{s}$ duration gate (0 schema rejections).
    3. **Editor Override Rate:** Frequency with which human video editors manually alter suggested trim boundaries.
  - *Statistical Sample Size & Power:*
    Use power analysis ($\alpha = 0.05, 1 - \beta = 0.80$) to determine minimum detectable effect (MDE) for a 5% relative increase in 3s view rate before launching.

---

## Section 3: Interviewer Follow-Up Chains (7-Step Deep Dives)

### Chain DA-1: Data Ingestion, Precision, and Schema Evolution

```
[Step 1: Conceptual] ──► Why integer centiseconds?
        │
[Step 2: Implementation] ─► How does regex handle missing minutes or milliseconds?
        │
[Step 3: Design Reasoning] ─► Why frozen dataclasses with slots instead of Pydantic in Phase 1?
        │
[Step 4: Trade-off] ────► Why not store raw floats and round at the presentation layer?
        │
[Step 5: Edge Cases] ───► What if two cues share the exact same millisecond start time?
        │
[Step 6: Optimization] ─► How does parser performance scale with 10,000 scripts?
        │
[Step 7: Production] ───► How do you version the JSON AST schema across seasons?
```

1. **Interviewer:** *"Why did you choose integer centiseconds instead of standard Python floats or integers representing milliseconds?"*
   - **Candidate:** *"Microdrama screenplays in our corpus define boundaries to two decimal places (`MM:SS.ss`). Centiseconds map 1:1 to hundredths of a second ($1\text{s} = 100\text{cs}$), avoiding floating-point inaccuracies ($0.1 + 0.2 = 0.30000000000000004$) while keeping integers small and human-interpretable without the memory bloat of nanosecond timestamps."*
2. **Interviewer:** *"Look at `parse_timestamp_to_cs`. How does it handle a timestamp that specifies 3 decimal digits, e.g., `00:14.355`?"*
   - **Candidate:** *"The regex `(?:\.(\d{1,3}))?` captures 1 to 3 digits. If length is 3 (milliseconds), line 159 executes `round(int(cs_raw) / 10.0)`, rounding milliseconds to the nearest centisecond (`35.5 -> 36cs`). If 1 digit (`00:14.3`), line 155 multiplies by 10 (`30cs`)."*
3. **Interviewer:** *"Why did you use standard library `@dataclass(slots=True, frozen=True)` in `parser.py` instead of Pydantic v2 models, which you used downstream in Phase 5?"*
   - **Candidate:** *"In Phase 1, `parser.py` instantiates 3,727 `ScreenplayElement` objects. Python dataclasses with `slots=True` allocate memory as a fixed-size C array, eliminating the per-instance `__dict__` overhead. This makes object creation ~4–5x faster and cuts memory consumption by ~60% compared to Pydantic, which does extensive runtime type validation better suited for API boundaries."*
4. **Interviewer:** *"Why not just keep floats in memory and round them using `round(val, 2)` only when filtering?"*
   - **Candidate:** *"Rounding floats in Python still produces binary floats, meaning `round(0.1 + 0.2, 2)` can still introduce subtle representation discrepancies in edge comparisons. Furthermore, repeatedly calling `round()` across 3,727 elements in multiple filter stages adds CPU overhead. Integer operations are native, exact, and faster."*
5. **Interviewer:** *"What happens in your parser if two consecutive cues have the exact same start and end timestamps (`00:10.00` to `00:10.00`)?"*
   - **Candidate:** *"In `ScreenplayElement.__post_init__`, line 70 enforces: `if self.end_cs <= self.start_cs: raise ValueError(...)`. A cue with zero duration violates strict monotonicity and will immediately throw a `ValueError`. If simultaneous multi-track elements exist (e.g. camera and dialogue occurring at the same time window `10.00` to `14.00`), both are valid and stored as distinct sequential elements in the scene's tuple."*
6. **Interviewer:** *"How would you optimize `ast_builder` if we needed to parse 100,000 episode scripts nightly in an automated ingestion pipeline?"*
   - **Candidate:** *"First, replace `finditer` regex on entire multi-kilobyte strings with a streaming compiled finite-state machine (or generator yielding lines). Second, utilize `multiprocessing.Pool` or Ray to distribute parsing across CPU cores, since screenplay parsing is an embarrassingly parallel, CPU-bound task ($O(N)$ across independent files). Third, replace JSON disk serialization with Apache Arrow / Parquet for columnar compression."*
7. **Interviewer:** *"Suppose writers introduce a new element type next season, such as `**ANIMATION:**` or `**VFX:**`. How does your schema versioning prevent downstream pipeline breakage?"*
   - **Candidate:** *"In `ElementType`, we define an `OTHER` or `VFX` enum. In `decompose_tag`, unknown tags default to `ElementType.ACTION` or `OTHER` rather than crashing. For schema versioning, the exported JSON AST includes a top-level `"schema_version": "1.1.0"`. Downstream modules validate the version string before processing."*

---

### Chain DA-2: Measuring Classification Accuracy and Error Analysis

```
[Step 1: Conceptual] ──► What metrics evaluate rule-based text classification?
        │
[Step 2: Implementation] ─► How do you construct the ground truth test corpus?
        │
[Step 3: Design Reasoning] ─► Why Precision vs Recall trade-off matters for Dialogue?
        │
[Step 4: Trade-off] ────► Macro-F1 vs Micro-F1 across imbalanced element classes?
        │
[Step 5: Edge Cases] ───► Ambiguous cues: "GERALD (O.S.)" vs "SOUND FX: DOOR SLAM"?
        │
[Step 6: Optimization] ─► How do you locate the exact regex causing false positives?
        │
[Step 7: Production] ───► Automated regression testing for script dialect changes.
```

1. **Interviewer:** *"How would you measure the classification accuracy of your screenplay parser across Camera, Action, Sound FX, and Dialogue?"*
   - **Candidate:** *"I would construct a multi-class confusion matrix against an annotated ground-truth test corpus and compute Class-Specific Precision, Recall, and F1-Scores, alongside Macro-Averaged F1 to account for severe class imbalance (Camera $47\%$ vs Sound FX $3\%$)."*
2. **Interviewer:** *"How would you create that ground-truth corpus without manually annotating 50,000 lines of script?"*
   - **Candidate:** *"I would use Stratified Random Sampling: sample 50 scenes across early, middle, and late episodes. Have two independent annotators label each element, measure Inter-Annotator Agreement using Cohen's Kappa ($\kappa$). Elements with $\kappa \ge 0.85$ form the gold-standard evaluation set."*
3. **Interviewer:** *"Between Precision and Recall for `DIALOGUE`, which metric is more critical for our downstream video promo discovery engine, and why?"*
   - **Candidate:** *"Precision for Dialogue is critical. If an `ACTION` description is misclassified as `DIALOGUE`, the system attempts to quote it as spoken dialogue in `TrimRecommendation.exact_opening_dialogue` (`"Juliana turns around with tears in her eyes"`), creating nonsensical ad captions. High precision ensures every dialogue cue is genuinely spoken dialogue."*
4. **Interviewer:** *"Why choose Macro-F1 over Micro-F1 or overall Accuracy?"*
   - **Candidate:** *"In our corpus of 3,727 elements, Sound FX accounts for only 114 elements ($3.0\%$). If a broken parser completely fails to detect Sound FX, overall accuracy would still exceed $96.9\%$, and Micro-F1 would hide the failure. Macro-F1 weights each class equally, immediately exposing a collapse in rare class detection."*
5. **Interviewer:** *"What is the most common classification ambiguity you encountered in screenplay parsing?"*
   - **Candidate:** *"The ambiguity between an off-screen character dialogue cue like `GERALD (O.S.): 'Stop!'` and a sound effect tagged as `SOUND FX (O.S.): 'Door slams'`. In `decompose_tag`, I check `any(s in tag_upper for s in ['SOUND FX', 'SFX', 'AUDIO'])` first. Only if no audio keyword matches do we inspect for character delivery and speaker names."*
6. **Interviewer:** *"If your validation audit reveals that Dialogue Recall dropped by 15% on a new season, how would you pinpoint the buggy regex pattern?"*
   - **Candidate:** *"I would run an error analysis script diffing failed extractions against the unparsed raw lines. Group unparsed lines by formatting patterns (e.g. counting colons, asterisks, brackets). If 80% of dropped lines lack a closing colon `**SPEAKER**` without `:`, it immediately pinpoints that `element_pattern` requires optional colon tolerance."*
7. **Interviewer:** *"How would you integrate this evaluation into continuous integration (CI/CD) before deploying parser updates?"*
   - **Candidate:** *"I would build a pytest regression suite asserting that on our 50-episode reference corpus, the parser produces exactly 3,727 elements with zero monotonicity failures. Any PR that alters element count or causes a regression in the golden fixture JSON fails the build."*
