# Class Distribution Analysis

nuScenes Mini dataset class distribution and its impact on model behavior.

---

## nuScenes Mini — Training Set (323 images, seed=42)

| Class | Instances | % of total | Notes |
|---|---|---|---|
| car | ~4,800 | ~52% | dominant class |
| pedestrian | ~2,100 | ~23% | second most common |
| barrier | ~900 | ~10% | static object |
| traffic_cone | ~600 | ~7% | static object |
| truck | ~300 | ~3% | similar to car |
| bicycle | ~120 | ~1% | rare |
| bus | ~100 | ~1% | rare |
| motorcycle | ~80 | ~1% | rare |

*Instance counts estimated from nuScenes Mini statistics.*

---

## Key Observations

### 1. Severe class imbalance

Car and pedestrian account for ~75% of all instances. Rare classes
(bicycle, bus, motorcycle) each have fewer than 150 instances in the
full training set. At 80/20 split, rare classes have ~100 training examples.

```
car        : ~4800 instances  → strong detector ✅
pedestrian : ~2100 instances  → good detector ✅
barrier    :  ~900 instances  → moderate detector ⚠️
traffic_cone: ~600 instances  → moderate detector ⚠️
bus        :  ~100 instances  → weak detector ❌
motorcycle :   ~80 instances  → weak detector ❌
bicycle    :  ~120 instances  → weak detector ❌
```

### 2. Imbalance explains observed behavior

In sanity check on bus.jpg (street-level photo):

```
ch[5] (pedestrian) max = 0.86  ← strong ✅
ch[4] (car)        max = 0.04  ← 4%, not detected
ch[8] (bus)        max = 0.007 ← 0.7%, not detected
```

The bus in bus.jpg is not detected because:
1. Bus has only ~100 training instances
2. bus.jpg is street-level, not dashcam perspective (out-of-distribution)
3. Combined effect: near-zero confidence

Pedestrians detected at 86%/72%/65% because:
1. 2100 training instances
2. Pedestrians in bus.jpg match nuScenes pedestrian appearance

### 3. Impact on mAP

mAP50 = 0.558 is a mean across all 8 classes. Per-class breakdown
expected to be roughly:

```
car        : mAP50 ~0.75-0.80 (high data, easy class)
pedestrian : mAP50 ~0.65-0.70
barrier    : mAP50 ~0.55-0.60
traffic_cone: mAP50 ~0.45-0.55
truck      : mAP50 ~0.40-0.50
bus        : mAP50 ~0.20-0.35 (low data)
bicycle    : mAP50 ~0.20-0.30 (low data)
motorcycle : mAP50 ~0.15-0.25 (low data)
```

*Per-class mAP not explicitly measured — training logs only report mean.*

---

## nuScenes Class Characteristics

nuScenes uses a different class taxonomy than COCO:

```
nuScenes              COCO equivalent
─────────────────────────────────────
car                 → car
pedestrian          → person
bus                 → bus
truck               → truck
bicycle             → bicycle
motorcycle          → motorcycle
traffic_cone        → (no equivalent)
barrier             → (no equivalent)
```

Traffic cones and barriers are nuScenes-specific classes not present
in COCO pretrained weights. Models fine-tuned from COCO checkpoints
must learn these classes from scratch, increasing data requirements.

---

## Mitigation Strategies (Production)

For a production system, class imbalance would be addressed by:

```
1. Data augmentation: copy-paste rare class instances
2. Focal loss: down-weight easy (car) examples
3. Class-weighted sampling: oversample rare classes
4. More data: nuScenes full (700 scenes vs 10 mini scenes)
   Full nuScenes: ~28k frames vs 404 in Mini
5. Transfer learning: use COCO pretrained backbone,
   fine-tune on nuScenes
```

For this project, imbalance is accepted as a known limitation.
The focus is deployment pipeline engineering, not production accuracy.