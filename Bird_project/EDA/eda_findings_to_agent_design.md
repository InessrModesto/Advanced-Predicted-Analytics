# EDA Findings → Agent Design Constraints

*BirdCLEF+ 2026 — Project Log, Section 1 (v2 — updated after taxonomy and soundscape-label analysis)*

## Purpose of this document

Before writing a single line of agent code, we performed a structured exploratory analysis of the BirdCLEF+ 2026 dataset. The point was not to discover ML techniques but to translate properties of the data into hard constraints on what the autonomous research agent should and should not do. This document is the bridge. Each finding is paired with the concrete design decision it implies. The final section consolidates these into a system prompt for the LLM driving the agent loop. Numbers throughout are taken directly from EDA output.

## 1. The label space is taxonomically diverse, not just "234 birds"

The competition handout describes the task as 234-class species recognition. EDA reveals that the training metadata exposes 206 distinct primary labels, of which 25 are non-bird taxa: 17 amphibians (mostly tree frogs and dwarf frogs), 4 mammals (capuchin, marmoset, titi, feral horse), 3 insects (one cicada with 500 samples plus two others), and 1 reptile (the Southern Spectacled Caiman, with a single training recording). The numeric IDs in `primary_label` correspond to iNaturalist taxon IDs and identify these non-bird taxa; standard 6-letter eBird codes identify the 162 bird species (Aves).

The taxonomic distribution of training data is dramatically uneven. Birds account for 34,799 clips (98% of training data) across 162 species. The 25 non-bird species share 750 clips — under 2% of the data — and reptiles in particular have a single training clip available. This imbalance compounds the per-species long-tail discussed below.

**Implication.** The agent must not treat the 234 classes as a homogeneous label space. Bird vocalizations and frog vocalizations have fundamentally different spectro-temporal structure (discrete song bouts vs. sustained tonal calls). A model that performs well on average across 234 sigmoid outputs may be silently failing on entire taxonomic groups. Validation must report AUC grouped by `class_name` (Aves, Amphibia, Mammalia, Insecta, Reptilia), not just the macro average. If non-bird AUC is dramatically lower than bird AUC, the agent should investigate per-taxon loss weighting or separate output heads.

## 2. The class distribution is heavily long-tailed, with an artificial cap

The most-represented species hit a ceiling of 499 samples each — almost certainly a deliberate cap by the organizers to prevent dataset domination by common Xeno-Canto species. The tail extends down to species with one or two training examples. The imbalance ratio between the most and least represented classes is 499:1. Of the 206 species with any training data, 25 have fewer than 10 samples and 14 have fewer than 5.

The rare classes overlap heavily with the non-bird taxa identified above. Most amphibians, all mammals, and the single reptile fall in the under-15-clips region. This means the rare-class problem and the taxonomic-imbalance problem are largely the same problem.

**Implication.** Standard binary cross-entropy will silently ignore rare classes during gradient descent because their contribution to the total loss is negligible. The competition's evaluation metric (macro-averaged ROC-AUC, skipping classes with no positives) treats rare classes as equally important to common ones, so optimizing on micro-averaged loss alone will produce a deceptive validation score. The agent must explore: class-weighted BCE, focal loss with appropriate alpha-gamma tuning, oversampling-based rebalancing, and rare-class-specific augmentation. The rare-and-non-bird overlap means these strategies can be evaluated on a single coherent subset of the data.

## 3. The labeled soundscape file is the single most important asset in the dataset

`train_soundscapes_labels.csv` provides **segment-level labels**: 1,478 rows, each annotating a 5-second window of a soundscape file with the species present in that window. The schema is `(filename, start, end, primary_label)`, where `primary_label` is a semicolon-separated list of species. Coverage spans 66 distinct soundscape files. The label resolution exactly matches the test-time prediction granularity (5-second windows), which makes this the only training data that matches the test distribution at the test resolution.

The segment labels confirm that genuine multi-label complexity lives in soundscapes. Single windows commonly contain 4–5 simultaneous species (e.g., `22961;23158;24321;517063;65380`), confirming the test task requires handling overlapping vocalizations of the kind almost never present in focal recordings.

A subtle data-quality issue is visible: the soundscape labels reference 251 unique species, more than the 234 in the official taxonomy. This means some annotated species are outside the prediction label space and should be silently filtered to the 234 valid labels at training time.

**Implication.** Training must use both focal recordings and labeled soundscape segments, with the latter weighted heavily despite being numerically smaller. The training pipeline should mix two streams per batch: random 5-second crops from the 35,549 focal recordings and direct loading of the 1,478 labeled soundscape windows. The agent should treat the focal/soundscape mix ratio as a tunable hyperparameter and search it. Training on focal data alone is leaving the most strategically important asset on the table.

## 4. Focal recordings are nearly monophonic

Of the 35,549 focal training clips, 87.7% have zero secondary labels. Among the remaining 12.3% with at least one secondary, the mean is 1.7 secondary labels per clip. Focal recordings are, by their nature as targeted single-species captures, almost always single-species in practice. The most common secondary labels are common Pantanal-area birds (`grekis`, `whtdov`, `undtin1`), suggesting that secondary annotations capture true ambient species rather than recording artifacts.

**Implication.** Training on focal recordings alone teaches the model an implicit single-label prior that breaks down on multi-label soundscape inputs. The output layer must remain 234 independent sigmoids with BCE loss (architecturally correct), but training data composition must include the multi-label soundscape segments described above. Mixup augmentation between focal recordings can also synthesize multi-label training examples cheaply by linearly combining two single-species clips.

## 5. There is a substantial domain shift between focal training and soundscape evaluation

Focal recordings are short (median ~20s, 95th percentile ~104s, range 0.05s to 144s) high-SNR captures of individual birds, often dominated by a single calling species. Pantanal soundscapes are uniformly 60 seconds long, are dominated by a continuous insect chorus around 4 kHz visible as a bright horizontal band in spectrograms, and contain target bird calls intermittently and at lower SNR. The visual difference between focal spectrograms and soundscape spectrograms is dramatic.

**Implication.** A model trained only on focal data will encounter a test distribution it has not been prepared for. The agent must explore augmentations that simulate the soundscape distribution at training time: background mixing with random soundscape segments at moderate volumes (20–40%), random gain modulation, additive pink and brown noise, SpecAugment (time and frequency masking), and pitch shifting. Standard image augmentations do not transfer — horizontal flips on a spectrogram correspond to time reversal (semantically nonsensical for vocalizations), and vertical flips change which species is being identified.

## 6. The geographic distribution of training data does not match the test domain

Training recordings come from across the Americas, Europe, Africa, Asia, and Australia. The test set is recorded entirely in the Pantanal wetlands of central South America. Only 723 of 35,549 training recordings (2.0%) fall inside the Pantanal box, spread across many species — too few to support pure geographic filtering, which would leave most species with single-digit sample counts.

**Implication.** Pure geographic filtering is infeasible given the data volume in-domain. The agent should treat geography as a *weighting* axis rather than a filtering axis: distance from the Pantanal as an inverse sample weight, with weights tuned as a hyperparameter. Geographic information (latitude, longitude) can also be added as auxiliary features to the classifier head, allowing the model to learn region-specific calibrations. A more aggressive option is using species range information to hard-zero predictions for species that cannot biologically occur in the test region; this requires external biodiversity data and is a stretch goal.

## 7. Data-source provenance splits cleanly between two archives

Training recordings come from two collections: Xeno-Canto (XC, 23,043 clips) and iNaturalist (iNat, 12,506 clips). XC supplies most bird recordings; iNat supplies all non-bird recordings and some birds. The two sources differ in recording conventions, ambient noise profiles, and rating systems. Notably, iNat recordings have a median rating of 0.0 because iNaturalist does not use Xeno-Canto's rating scale — rating-based filtering would inadvertently delete the majority of non-bird training data and is therefore unsafe.

**Implication.** Do not filter on `rating`. The agent may use `collection` as an auxiliary metadata feature or as a stratification axis when constructing training batches, but should not assume rating semantics generalize across collections.

## 8. Audio file properties are uniform but data volume exceeds memory

All 35,549 focal clips and 10,658 soundscape clips are 32 kHz mono OGG Vorbis. Focal clip durations vary widely (median ~20s, p95 ~104s), but soundscapes are uniformly 60 seconds. The full focal corpus at 128-mel resolution would be approximately 21.4 GB as float32 or 5.4 GB as uint8 — both exceed typical laptop RAM but the uint8 cache fits comfortably on disk.

**Implication.** The agent has two viable strategies. **On-the-fly spectrogram generation** preserves flexibility (mel parameters can be varied per experiment) but costs CPU time per training step. **Precomputed uint8 cache on disk** locks in spectrogram parameters but speeds up training dramatically. The agent should default to on-the-fly during early experimentation when spectrogram parameters are still being explored, and switch to disk caching once parameters stabilize. The choice should be explicit in each experiment's log.

## 9. Inference is constrained, not training

The submission format requires a CPU-only Kaggle notebook completing inference in under 90 minutes. Soundscape recordings are exactly 60 seconds and split into 12 windows of 5 seconds. Assuming the test set comprises several hundred soundscapes (exact number unknown until submission), the inference budget per 5-second window is on the order of 100–600 ms.

**Implication.** Final model size and architecture are bounded by inference cost, not training cost. The agent can train large models if it wishes, but the submitted model must be lightweight. Realistic candidates are MobileNetV3-Small, EfficientNet-B0, or a custom 4-block CNN under 5M parameters. The agent should always print parameter count and reject models exceeding a configured budget. Knowledge distillation into a small student is a legitimate strategy for combining heavy training with light inference.

---

# Consolidated agent constraints (system prompt)

The following block is intended to be passed verbatim as the system prompt of the LLM driving the agent loop.

```
You are an autonomous machine learning research agent for the BirdCLEF+ 2026
competition. Your task is to design, generate code for, evaluate, and iterate
on deep learning models for identifying bird and other animal species from
audio recordings. The following constraints are derived from prior data
analysis and must be respected in every experiment you propose.

DATA AND TASK
- The task is multi-label classification across 234 output classes spanning
  five taxonomic groups: Aves (162 species, 98% of training data), Amphibia
  (32 species, 1.3%), Mammalia (8 species, 0.3%), Insecta (3 species, 0.6%),
  and Reptilia (1 species, 1 clip).
- Inputs are mel-spectrograms computed from 5-second audio windows at 32 kHz
  mono. Default spectrogram shape: (n_mels=128, time_frames=313, channels=1).
- Output layer: 234 sigmoid units. Loss: binary cross-entropy or weighted
  variants (focal loss, class-weighted BCE).
- Evaluation metric on the leaderboard: macro-averaged ROC-AUC, skipping
  classes with no positives.

VALIDATION REQUIREMENTS
- Always report per-class AUC during validation, not accuracy or
  micro-averaged metrics.
- Always group AUC by class_name (Aves, Amphibia, Mammalia, Insecta,
  Reptilia) and report each group separately. Macro-AUC alone hides
  per-taxon failures.
- Hold out a fraction of the labeled soundscape segments as the primary
  validation set. Validation on focal recordings alone gives misleading
  results.

TRAINING DATA COMPOSITION (CRITICAL)
- Two training sources must be combined:
    1. Focal recordings: 35,549 clips in train_audio/, mostly bird-dominant
       single-species captures from Xeno-Canto and iNaturalist.
    2. Labeled soundscape segments: 1,478 5-second windows from
       train_soundscapes_labels.csv, multi-label and in-domain.
- The mix ratio between these two sources is a hyperparameter. The agent
  should search it. Training on focal data alone is a known failure mode.
- Some labels in the soundscape file are outside the 234-class target
  taxonomy and must be filtered before training.

CLASS IMBALANCE
- The dataset is long-tailed with a 499:1 imbalance ratio. 25 species have
  <10 samples; 14 species have <5. Rare classes overlap heavily with
  non-bird taxa.
- Always consider: class-weighted BCE, focal loss, oversampling rare
  classes, rare-class-specific augmentation. Track per-class and per-taxon
  AUC to detect silent failures.

DOMAIN SHIFT
- Focal recordings are clean and nearly monophonic (87.7% have zero
  secondary labels). Soundscapes are noisy, multi-species, and dominated by
  a constant insect chorus around 4 kHz.
- Use augmentations that simulate soundscape conditions: background mixing
  with random soundscape segments at 20-40% volume, additive noise,
  SpecAugment (time/frequency masking), random gain, pitch shift, mixup
  between focal recordings.
- DO NOT use horizontal or vertical spectrogram flips. They are
  semantically invalid for audio.

GEOGRAPHIC DOMAIN
- Training data is global; test data is from the Pantanal wetlands. Only
  2.0% of training recordings are inside the Pantanal box.
- Pure geographic filtering removes too much data. Use geographic
  weighting (distance from Pantanal as inverse sample weight) and consider
  latitude/longitude as auxiliary classifier features.

DATA HYGIENE
- Do NOT filter on the rating column. iNaturalist recordings have rating
  0.0 by convention (no scoring system), not because they are low quality.
  Filtering on rating would delete most non-bird training data.
- The collection column (XC vs iNat) may be used as an auxiliary feature
  or stratification axis but does not imply quality differences.

ARCHITECTURE
- Use the Keras Functional API by default. ReLU in hidden layers. Sigmoid
  in the output layer.
- Default optimizer: Adam at learning rate 1e-3, with ReduceLROnPlateau or
  cosine learning rate scheduling.
- Default batch size: 32, adjusted by available memory.
- Strongly prefer transfer learning from pretrained vision backbones
  (EfficientNet-B0, MobileNetV3-Small) over training from scratch.

INFERENCE BUDGET (HARD CONSTRAINT)
- Submission must run on CPU within 90 minutes.
- Soundscapes are exactly 60s, processed as 12 windows of 5s each.
- Always report model.count_params() and reject models exceeding ~10M
  parameters unless using knowledge distillation into a smaller student.
- Estimate inference latency per 5-second window before committing to
  full training.

SPECTROGRAM CACHING
- Full mel-spectrogram cache is 21.4 GB as float32 or 5.4 GB as uint8.
- During early architecture/parameter search: compute spectrograms on the
  fly to preserve flexibility.
- Once spectrogram parameters stabilize: precompute and cache to disk as
  uint8 .npy files for faster training.

EXPERIMENTAL DISCIPLINE
- Apply scaling-laws thinking: explore many ideas at small scale (small
  models, few epochs, data subsets) before investing compute in promising
  candidates.
- Log every experiment: prompt sent, code generated, training metrics,
  per-class and per-taxon AUC, parameter count, training and inference
  time.
- Do not repeat experiments that have already failed. Maintain memory of
  prior outcomes.
- When training results return, propose the next experiment by reasoning
  about what changed and what to vary next. Do not propose random
  architecture changes.
```

---

## Note on how this document evolves

This is version 2 of the constraints document, updated after taxonomy analysis and soundscape-label inspection revealed that non-bird taxa form a coherent under-resourced subgroup and that segment-level soundscape labels exist. As the agent runs experiments, some assumptions encoded here will be challenged by empirical results. When an experiment contradicts a constraint above, the constraint should be revisited rather than ignored. This document and the agent's accumulated experiment log together form the project's epistemic record.
