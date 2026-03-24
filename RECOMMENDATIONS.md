# Recommendations For The Next Project Phase

This document focuses on what can be done next, especially when new CVAT exports from three different teammates will be added and the project grows beyond the current carton-only dataset.

## Current Position

The project is already in a good intermediate state:

- one custom CVAT dataset was integrated
- the dataset can be inspected and cleaned
- a custom YOLO model was trained
- inference works with simple launcher commands
- export flow exists for edge deployment preparation

That means the next work should not be random feature growth. It should focus on scaling the dataset and model lifecycle cleanly.

## Highest Priority Recommendations

### 1. Standardize All Future CVAT Exports

When three different teammates provide CVAT exports, the biggest risk is inconsistency.

Typical problems:

- different class names for the same object
- spelling differences such as `carton`, `Karton`, `milk_carton`, `Milch_Karton`
- different image naming conventions
- different label quality
- missing labels or empty annotations
- different train/val/test assumptions

Recommended rule:

Create one shared class dictionary before accepting new datasets.

Example:

```text
0: Milk_Carton_Chocolate
1: Milk_Carton_Vanilla
2: Tea_Box
3: Cube_Carton
4: Glass
5: Metal_Can
6: Plastic
```

If teammates use different names, map them into this canonical schema before training.

Best next implementation idea:

- add a `merge-datasets` command
- read several CVAT YOLO exports
- validate each one
- remap class names and IDs into one unified label space
- produce one merged training dataset

### 2. Introduce Dataset Versioning

As soon as several teammates deliver exports, the project needs dataset versions.

Recommended structure:

```text
data/
  cvat_exports/
    teammate_a_v1/
    teammate_b_v1/
    teammate_c_v1/
  merged/
    recycling_dataset_v1/
    recycling_dataset_v2/
```

Every merged dataset version should save:

- source datasets used
- class mapping used
- number of images
- number of labels
- missing labels removed
- train/val/test split
- timestamp

This is important because later you will ask:

- which data produced the best model
- which teammate export introduced noise
- whether accuracy improved after adding more data

### 3. Separate Task Types Early

Right now the project is carton-focused. Later the problem may become general recycling detection.

That changes the strategy.

There are two possible directions:

1. narrow model
This means one custom model specialized for your own recycling classes.

2. hybrid model
This means combining:
- existing pretrained object categories when they already work
- custom fine-tuned classes for domain-specific classes that pretrained models do not understand well

For example:

- if a pretrained model already detects `bottle`, `cup`, or `can` well enough, you may reuse that
- but custom packaging classes such as specific cartons or local recycling categories may still need fine-tuning

## Recommendation For Future Teammate CVAT Data

When new CVAT exports arrive, I would strongly recommend this workflow:

1. put each export into its own folder
2. inspect each dataset separately
3. generate a validation report per dataset
4. review class names and class balance
5. map classes into a shared canonical schema
6. build one merged dataset
7. train one merged model
8. compare results against the previous best model

This avoids mixing bad annotations into training blindly.

## Recommendation About Pretrained Recycling Models

You asked an important question: maybe there are already YOLO or similar models trained for metal, glass, carton, and related classes.

That is a valid option, but it should be evaluated carefully.

### Good use cases for pretrained models

- when classes are generic and already well represented in public datasets
- when you need a fast baseline
- when your own dataset is still small
- when edge deployment speed matters more than perfect domain fit

### Risks of pretrained models

- class definitions may not match your recycling categories
- local packaging may look different from the data used in the pretrained model
- public waste datasets may use labels that are too broad
- performance on Raspberry Pi may still require export and optimization work

### Practical recommendation

Do not replace your custom pipeline with a pretrained model immediately.

Instead:

1. keep your custom training pipeline as the main backbone
2. test 1 or 2 public pretrained recycling or waste-sorting models as baselines
3. compare them against your own custom model on the same validation images

If a pretrained model performs surprisingly well, it can become:

- the base model for fine-tuning
- or a fallback model for broader categories

## Best Near-Term Technical Additions

If I were continuing this project, these would be the strongest next features.

### A. Dataset Merge Utility

Add a module such as:

```text
yolo_edge/data/dataset_merger.py
```

It should:

- load multiple CVAT YOLO exports
- validate each dataset
- merge images and labels
- remap class IDs
- write one canonical `data.yaml`
- output a merge report

This would be one of the highest-value additions for your multi-person workflow.

### B. Evaluation Command

Add a dedicated evaluation workflow:

```text
main.py evaluate
```

It should produce:

- precision and recall per class
- mAP values
- confusion matrix
- failure case examples
- class imbalance report

That becomes very important once several datasets and several candidate models exist.

### C. Model Registry Convention

Instead of storing only `best.pt`, save structured model metadata.

Recommended structure:

```text
runs/models/
  recycling_v1/
    best.pt
    labels.txt
    training_summary.yaml
    dataset_report.yaml
    export_manifest.yaml
```

This makes team collaboration much easier.

### D. Edge Benchmark Command

For Raspberry Pi 5 and AI camera work, add:

```text
main.py benchmark-edge
```

It should measure:

- average latency
- max latency
- FPS
- CPU load
- RAM use
- temperature if available

This matters more on the Pi than on the laptop.

## Suggested Strategy For The Next 3 Team Datasets

Recommended sequence:

1. collect all three CVAT exports without editing them
2. inspect and report each one independently
3. define one shared label taxonomy
4. remap all three datasets into the shared taxonomy
5. merge them into one versioned dataset
6. train a merged baseline model
7. compare against the current carton model
8. export the best model for Raspberry Pi

This is safer than training directly on mixed raw exports.

## Suggested Class Planning

If the project expands toward recycling, define classes by operational need, not only by visual difference.

Example possible recycling taxonomy:

- `Carton`
- `Glass`
- `Metal`
- `Plastic`
- `Paper`
- `Tea_Box`
- `Milk_Carton`
- `Can`

Then decide whether you want:

- coarse sorting classes
or
- fine-grained packaging classes

Do not mix both strategies accidentally in the same dataset without a clear plan.

## Recommended Decision About Model Direction

My recommendation would be:

1. keep the current custom YOLO pipeline as the core system
2. add multi-dataset merge support next
3. test one or two public pretrained waste or recycling models as baselines
4. fine-tune the best starting point on your merged dataset
5. optimize the final model specifically for Raspberry Pi 5 deployment

That gives you both flexibility and control.

## Concrete Next Steps

If we continue development, I would do the following next:

1. implement `dataset_merger.py`
2. implement class remapping from multiple CVAT exports
3. add `evaluate` command
4. add `benchmark-edge` command
5. add a model registry folder structure
6. test a public recycling-pretrained model against your custom model

## Final Recommendation

The project is already beyond prototype stage. The next success factor is no longer only "can YOLO detect something?" but rather:

- can several people contribute datasets consistently
- can models be compared reliably
- can the final model run efficiently on Raspberry Pi 5

That is why the next best investment is dataset governance and evaluation, not only more training.
