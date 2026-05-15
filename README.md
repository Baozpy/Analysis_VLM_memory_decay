# Analysis_VLM_memory_decay

This repository contains the experimental pipeline and analysis code for studying **memory decay and error propagation in Vision-Language Models (VLMs)** during multi-turn, multi-image dialogues.

The project focuses on a key question:

> Do Vision-Language Models gradually forget earlier visual and textual context as a conversation becomes longer, and do early mistakes affect later answers?

To answer this question, this project builds a multi-model evaluation framework, tests several open-source VLMs on multi-turn multimodal dialogue data, and analyzes model behavior using turn-level accuracy, decay slopes, revisit-based measurements, and qualitative case studies.

---

## 1. Project Motivation

Modern Vision-Language Models can answer questions about images and follow multi-turn conversations. However, in realistic dialogue settings, users often refer back to earlier images, previous answers, or objects mentioned several turns ago.

In these cases, VLMs may show several problems:

- They may forget information from earlier turns.
- They may focus too much on recent context.
- They may carry forward wrong assumptions from previous answers.
- They may hallucinate visual details that were not present in the image.
- They may perform well on single-turn questions but become unstable in longer conversations.

This project aims to quantify these issues instead of only describing them qualitatively.

---

## 2. Research Questions

This project studies the following questions:

1. **Turn-level memory decay**  
   Does model accuracy decrease as the dialogue progresses?

2. **Position and recency bias**  
   Are models more accurate when the relevant image or information appears closer to the current question?

3. **Error propagation**  
   If a model makes a mistake in an earlier turn, does that mistake influence later answers?

4. **Revisit behavior**  
   When a question refers back to an earlier image or entity, can the model correctly retrieve and use the old context?

5. **Model comparison**  
   Do different VLMs show different memory-decay patterns?

---

## 3. Models

The current pipeline supports experiments with multiple Vision-Language Models, including:

- **BLIP-2**
- **LLaVA**
- **Qwen-VL / Qwen2-VL**

The code is designed so that additional VLMs can be added by implementing new model wrapper functions.

---

## 4. Dataset

The project is mainly designed for multi-turn, multi-image dialogue datasets.

The main dataset used in development is:

- **MMDU**: Multi-Turn Multi-Image Dialog Understanding

The dataset contains dialogue samples where each conversation may include multiple images and multiple rounds of questions. This makes it suitable for studying how models handle visual memory across turns.

Because some datasets and image files may be large or have separate licenses, this repository may not include all raw image files directly. Please follow the dataset provider's instructions to download the original data.

---

## 5. Core Metrics

This project uses three main quantitative measurements.

---

### B1: Turn-Level Accuracy Curve

B1 measures model accuracy at each dialogue turn.

For each turn \(t\), the accuracy is:

\[
Acc(t) = \frac{1}{N_t} \sum_i \mathbf{1}(y_{i,t} = \hat{y}_{i,t})
\]

where:

- \(N_t\) is the number of samples at turn \(t\)
- \(y_{i,t}\) is the ground-truth answer
- \(\hat{y}_{i,t}\) is the model prediction
- \(\mathbf{1}(\cdot)\) is an indicator function

This metric helps show whether a model becomes worse as the conversation gets longer.

---

### B2: Accuracy Decay Slope

B2 measures the slope of model performance as dialogue turns increase.

The main idea is to fit a regression model:

\[
Y_{i,t} \sim \text{Turn}_{i,t}
\]

where \(Y_{i,t}\) is a binary correctness label.

A negative slope suggests that the model's accuracy decreases over longer dialogue contexts.

When possible, the project uses a mixed-effects or robust regression-style analysis. If that fails, a weighted least squares fallback is used.

The output includes:

- estimated slope
- standard error
- z-score

The z-score is computed as:

\[
Z = \frac{\hat{\beta}_1}{SE(\hat{\beta}_1)}
\]

A large negative z-score suggests stronger evidence of memory decay.

---

### D: Revisit / Distance-Based Memory Decay

D measures how well a model handles questions that refer back to earlier images or earlier dialogue turns.

The key idea is that accuracy may decrease as the distance between the current question and the relevant previous context increases.

A simple exponential decay form is used:

\[
Acc(\Delta) \approx A e^{-\lambda \Delta}
\]

where:

- \(\Delta\) is the distance from the current turn to the referenced earlier context
- \(A\) is the initial accuracy level
- \(\lambda\) is the decay rate

The half-life is:

\[
t_{1/2} = \frac{\ln 2}{\lambda}
\]

A shorter half-life means the model forgets earlier context more quickly.

---

## 6. Qualitative Analysis

In addition to numerical metrics, the project also includes qualitative case studies.

These examples help explain why a model fails. Common failure patterns include:

- confusing image identities
- answering based on the wrong image
- relying on recent turns instead of the relevant earlier turn
- repeating a previous hallucination
- changing an earlier answer without new evidence
- ignoring explicit image references such as “the first image” or “Image 2”

Qualitative examples are important because accuracy curves alone cannot fully explain model behavior.

---

## 7. Repository Structure

A possible structure of this repository is:

```text
Analysis_VLM_memory_decay/
│
├── README.md
├── requirements.txt
├── scripts/
│   ├── run_experiment.py
│   ├── evaluate_predictions.py
│   ├── compute_b1_turn_accuracy.py
│   ├── compute_b2_decay_slope.py
│   ├── compute_d_revisit_decay.py
│   └── make_figures.py
│
├── models/
│   ├── blip2_wrapper.py
│   ├── llava_wrapper.py
│   └── qwen_wrapper.py
│
├── data/
│   ├── README.md
│   └── sample/
│
├── results/
│   ├── raw_outputs/
│   ├── processed/
│   └── figures/
│
├── notebooks/
│   └── analysis.ipynb
│
└── figures/
    ├── b1_turn_accuracy.png
    ├── b2_decay_slope.png
    └── d_revisit_decay.png






## 8. Installation

Create a new Python environment:

    conda create -n vlm_memory python=3.10
    conda activate vlm_memory

Install dependencies:

    pip install -r requirements.txt

Common dependencies include:

    torch
    transformers
    accelerate
    datasets
    pandas
    numpy
    scikit-learn
    statsmodels
    matplotlib
    tqdm
    Pillow

Depending on the model, additional dependencies may be required.

---

## 9. Running Experiments

A typical workflow is shown below.

### Step 1: Prepare the Dataset

Download the dataset and place the image files under the expected directory.

Example structure:

    data/
    ├── mmdu/
    └── mmdu_pics/

If the dataset stores image paths as absolute paths or outdated relative paths, update the path-mapping function in the data loading script.

For example, some MMDU image paths may need to be mapped from:

    /mmdu_pics/...

to a local directory such as:

    ./_mmdu_pics/...

or:

    data/mmdu_pics/

---

### Step 2: Run Model Inference

Example for LLaVA:

    python scripts/run_experiment.py \
      --model llava \
      --data_path data/mmdu \
      --image_dir data/mmdu_pics \
      --output_path results/raw_outputs/llava_outputs.json

Example for BLIP-2:

    python scripts/run_experiment.py \
      --model blip2 \
      --data_path data/mmdu \
      --image_dir data/mmdu_pics \
      --output_path results/raw_outputs/blip2_outputs.json

Example for Qwen-VL / Qwen2-VL:

    python scripts/run_experiment.py \
      --model qwen \
      --data_path data/mmdu \
      --image_dir data/mmdu_pics \
      --output_path results/raw_outputs/qwen_outputs.json

The exact model name and checkpoint path may need to be adjusted depending on the local environment.

---

### Step 3: Evaluate Predictions

After model inference, evaluate the raw predictions against the ground-truth answers:

    python scripts/evaluate_predictions.py \
      --prediction_path results/raw_outputs/llava_outputs.json \
      --output_path results/processed/llava_eval.csv

The evaluation output should include at least:

- sample ID
- dialogue turn
- image reference
- question
- ground-truth answer
- model prediction
- correctness label
- model name

---

### Step 4: Compute B1 Turn-Level Accuracy

B1 measures how accuracy changes across dialogue turns.

    python scripts/compute_b1_turn_accuracy.py \
      --input results/processed/llava_eval.csv \
      --output results/processed/llava_b1.csv

The output can be used to plot the turn-level accuracy curve.

---

### Step 5: Compute B2 Decay Slope

B2 measures whether the model has a meaningful performance trend as the dialogue becomes longer.

    python scripts/compute_b2_decay_slope.py \
      --input results/processed/llava_eval.csv \
      --output results/processed/llava_b2.csv

The output usually includes:

- slope estimate
- standard error
- z-score
- p-value, if available
- model name
- recap or window setting, if used

A more negative slope usually means stronger performance decay.

---

### Step 6: Compute D Revisit / Distance-Based Decay

D measures how well the model handles questions that refer back to earlier images or earlier dialogue turns.

    python scripts/compute_d_revisit_decay.py \
      --input results/processed/llava_eval.csv \
      --output results/processed/llava_d.csv

This analysis focuses on the distance between the current question and the earlier visual or textual context that the model needs to recall.

---

### Step 7: Generate Figures

Generate the main figures:

    python scripts/make_figures.py \
      --input_dir results/processed \
      --output_dir results/figures

The expected figures include:

- B1 turn-level accuracy curve
- B2 decay slope or z-score figure
- D revisit / distance-based decay curve
- optional qualitative case-study figures

---

## 10. Main Outputs

The main outputs of this project are saved under the `results/` and `figures/` folders.

### B1 Figure: Turn-Level Accuracy

The B1 figure shows model accuracy at each dialogue turn.

This figure is used to answer:

> Does the model become less accurate as the conversation becomes longer?

A downward trend may suggest turn-level memory decay.

---

### B2 Figure: Decay Slope

The B2 figure summarizes the estimated decay slope across turns.

This figure is used to answer:

> Is there a statistically meaningful downward trend in model performance?

The B2 result usually includes a slope estimate and a z-score.

A negative slope means that accuracy tends to decrease as the turn index increases.

A larger absolute z-score means stronger evidence for the trend.

---

### D Figure: Revisit / Distance-Based Decay

The D figure shows model accuracy as a function of distance from the relevant earlier context.

This figure is used to answer:

> When the model needs to refer back to earlier images or earlier turns, does it perform worse as the distance becomes larger?

This metric is especially useful for studying visual memory in long conversations.

---

### Qualitative Case Studies

Qualitative examples are used to explain the failure patterns behind the numerical results.

Common patterns include:

- answering based on the wrong image
- confusing image order
- focusing too much on recent turns
- forgetting an earlier visual detail
- repeating a hallucinated detail
- changing an earlier answer without new visual evidence
- failing to follow references such as “the first image” or “Image 2”

These examples help connect the quantitative metrics with actual model behavior.

---

## 11. Example Findings

Some common patterns observed in this project include:

- VLMs may perform well in early turns but become less stable in later turns.
- Some models show clear recency bias.
- Models may rely more on the most recent image or question, even when the current question refers to an earlier image.
- Earlier mistakes can influence later answers.
- Revisit questions are harder when the relevant image appeared many turns earlier.
- Different VLMs show different memory-decay patterns under the same dialogue setting.
- A model may have high single-turn ability but still struggle with long multi-turn visual memory.

These findings suggest that multi-turn multimodal evaluation should not only report final accuracy. It should also measure how model behavior changes over the course of a conversation.

---

## 12. Why This Project Matters

Many real-world AI assistants need to handle long conversations involving images, screenshots, documents, or other visual inputs.

Examples include:

- visual question answering assistants
- educational tutoring systems
- shopping assistants
- medical image discussion tools
- robotics and embodied AI agents
- personal AI assistants with visual memory
- multimodal agents that interact with users over long sessions

In these settings, the model must not only understand the current image or question. It also needs to remember earlier visual information and use it correctly later.

If a model forgets earlier context or carries forward a wrong assumption, it may produce unreliable answers.

Therefore, understanding VLM memory decay is important for building more reliable multimodal systems.

---

## 13. Limitations

This project is an analysis-oriented research prototype. It has several limitations.

First, the dataset size may be limited by available compute resources. Running multiple large VLMs over multi-turn, multi-image conversations can be expensive.

Second, different models require different prompt formats and image preprocessing methods. This makes direct comparison harder.

Third, automatic correctness evaluation may not capture all acceptable answers, especially when the answer is open-ended.

Fourth, multi-turn visual memory is difficult to isolate from general reasoning ability. A model may fail because it forgets earlier context, but it may also fail because the question itself requires complex reasoning.

Fifth, different VLMs have different context length limits, image input limits, and chat-template requirements.

Because of these limitations, the results should be interpreted as analysis evidence rather than a final benchmark score.

---

## 14. Future Work

Possible future directions include:

- adding more VLMs
- testing longer conversations
- testing more multi-image dialogue datasets
- comparing different recap strategies
- studying whether summary-based memory reduces decay
- studying hallucination persistence across turns
- building a controlled benchmark specifically for visual memory decay
- adding retrieval-based memory for long multimodal conversations
- comparing open-source VLMs with closed-source multimodal models
- improving automatic evaluation with human annotation
- analyzing failure cases by question type, image position, and reference distance

---

## 15. Citation

If you use this repository or build on this work, please cite it as:

    @misc{yan2026vlmmemorydecay,
      title        = {Analysis of Memory Decay and Error Propagation in Vision-Language Models},
      author       = {Baozan Yan},
      year         = {2026},
      howpublished = {GitHub repository},
      note         = {Analysis_VLM_memory_decay}
    }

---

## 16. Author

**Baozan Yan**  
M.S. in Computer Science  
University of Maryland, College Park

Research interests:

- Vision-Language Models
- Multimodal Dialogue
- Long-Context Evaluation
- Model Memory and Forgetting
- Error Propagation
- NLP and Model Evaluation

---

## 17. License

This repository is intended for academic and research purposes.

Please check the licenses of the datasets, model checkpoints, and third-party code used in this project before redistribution or commercial use.

Suggested license:

    MIT License

However, if the repository depends heavily on datasets or model checkpoints with restricted licenses, please make sure the final license is compatible with those resources.
