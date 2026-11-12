# Human Activity Recognition: Tennis Stroke Classification

## Mini project using the THETIS dataset

This project compares two methods for recognizing 12 tennis strokes: classifying still images with ResNet50 and classifying body-pose sequences extracted from video.

| Field | Details |
|---|---|
| **Course project** | Human Activity Recognition |
| **Dataset** | THETIS RGB tennis videos |
| **Team** | K. Anjan Reddy (23BCS071), Babhale Rohan Laxmikant (23BCS026), Kshitij Shende (23BCS075) |
| **Guide** | Dr. Sunil C.K. |

This is a student project. The results describe the THETIS videos and the test split used here. They do not show how the models would perform in real tennis matches.

---

## 1. Project summary

The project has two pipelines:

1. **ResNet50 photo pipeline:** classifies one middle frame or three sampled frames from each video. For the three-frame experiment, predictions from the three still images were also averaged to produce one prediction per video.
2. **Skeleton pipeline:** YOLO11m-pose detects 17 body points in each video frame. After cleaning, each video is represented by 64 poses. Model A uses the middle pose; Model B uses the full sequence in time order.

Both pipelines use the same split by player. A player’s videos appear in only one of the training, validation, or test sets.

---

## 2. Dataset and submission contents

The project uses 1,980 RGB clips of 12 tennis strokes performed by 55 people: 31 beginners and 24 experienced players. The clips were recorded indoors. The ball is not visible, which makes some similar strokes difficult to distinguish.

The raw dataset is **not included** in this submission because it is large. Download it from the [THETIS dataset GitHub repository](https://github.com/THETIS-dataset/dataset). The project uses the repository’s `VIDEO_RGB` folder. To regenerate the image and pose data, place `VIDEO_RGB` under `dataset/VIDEO_RGB/` before running the extraction scripts.

The subject split was created with seed 42 and used in both pipelines:

| Split | Players | Videos |
|---|---:|---:|
| Train | 39 | 1,404 |
| Validation | 8 | 288 |
| Test | 8 | 288 |

The split mapping is in `manifests/split_manifest.csv`.

The submission folder is organized as follows. The raw `dataset/` folder is not present; its GitHub link is above.

```text
Mini_Project_Submission/
├── README.md
├── data_split/                       # One middle image per video
├── data_split_3frame/                # Three sampled images per video
├── manifests/                        # Split map and frame extraction logs
├── models/
│   └── best_model.pt                 # Best saved ResNet50 checkpoint
├── report/
│   ├── report.pdf
│   ├── main.tex
│   ├── references.bib
│   └── figures/
├── results/
│   └── results/                      # ResNet50 metrics, logs, and confusion matrices
├── skeleton_clean/
│   ├── <stroke_class>/               # 12 class folders containing cleaned pose files
│   └── skeleton_clean_report.csv
├── skeleton_final_tested/
│   ├── HAR_using_Gait.ipynb
│   ├── models_skeleton/              # Final Model A, Model B, and shuffled-control weights
│   └── results_skeleton/             # Final test results and confusion matrices
├── src/                              # ResNet50 source scripts
└── src_skeleton/                     # Skeleton pipeline source scripts
```

For the skeleton scripts to use the included cleaned poses, the 12 class folders should be directly inside `skeleton_clean/`. There should not be another `skeleton_clean/` folder between the root and the class folders.

---

## 3. ResNet50 photo pipeline

The one-frame experiment uses the middle frame of each video. The three-frame experiment samples an early, middle, and late frame. Each frame is first classified separately. The project also averages the three frame prediction scores to make one prediction for each video. This averaging does not make the model learn movement over time.

ResNet50 starts with weights trained on ImageNet. Training first updates the new 12-class output layer, then fine-tunes the last residual block and output layer. Validation data is used to select model settings and checkpoints; the held-out test set is used for final evaluation.

| Script | Purpose |
|---|---|
| `src/extract_frames.py` | Saves the middle frame from each video. |
| `src/extract_frames_3frame.py` | Saves three sampled frames from each video. |
| `src/make_split.py` | Creates the subject-wise split manifest. |
| `src/materialize_split.py` and `src/materialize_split_3frame.py` | Arrange extracted images into split and class folders. |
| `src/verify_split_integrity.py` and `src/verify_split_integrity_3frame.py` | Check that subjects do not appear in more than one split. |
| `src/dataset.py` | Loads and prepares image batches. |
| `src/model.py` | Builds the pretrained ResNet50 classifier. |
| `src/train.py` | Contains the two-phase training routine. |
| `src/evaluate.py` | Calculates test metrics and creates a confusion matrix. |
| `src/spotcheck_3frame_sampling.py` | Checks examples from three-frame sampling. |

### Photo pipeline results

| Experiment | Test accuracy | Evaluation unit |
|---|---:|---|
| One middle frame | 30.56% | 288 videos |
| Three frames scored individually | 22.69% | 864 images |
| Three frame predictions averaged per video | 33.33% | 288 videos |

The 22.69% result is measured over images, while the other two are measured over videos.

---

## 4. Skeleton pipeline

### Pose preparation

`phase1_extract_skeletons.py` runs YOLO11m-pose on the RGB videos. It saves joint coordinates, confidence scores, and person bounding boxes in raw pose files.

`phase2_skeleton_qc.py` checks detection quality and writes a report without changing the data. `phase2_reextract_flagged.py` can reprocess selected videos and try to follow the same person across frames. The new detections are saved separately for review.

`phase2_clean_skeletons.py` marks unreliable frames, estimates missing joint positions from nearby frames, centers poses on the midpoint between the hips, and scales them using torso length. It then resamples each video to 64 evenly spaced time steps. The script saves cleaned poses in `skeleton_clean/` and writes a cleaning report.

### Models and evaluation

- **Model A** uses one pose from the middle of the swing.
- **Model B** uses all 64 poses in their original time order.
- **Shuffled control** is trained on sequences whose pose order has been shuffled.

`phase3_4_train_skeleton.py` runs an initial training of Models A and B using training and validation data only. `phase4b_skeleton_checks.py` compares settings using validation data and five seeds; it does not use the test set. The selected final setups are trained again in `phase5_skeleton_test_once.py`, then evaluated on the held-out test players.

`phase5_finish_outputs.py` saves reports and confusion matrices from predictions already made if the final evaluation cell was interrupted. It does not train again or rerun the test evaluation. `view_skeleton.py` and `View_Skeleton.ipynb` display stick figures, animations, or video overlays.

### Skeleton results

| Setup | Mean test accuracy across five seeds |
|---|---:|
| Model A: one middle pose | 43.6% ± 1.1% |
| Shuffled-sequence control | 56.9% ± 1.8% |
| Model B: ordered sequence | 68.0% ± 1.4% |

In a separate check, the already-trained ordered Model B was tested after shuffling the pose order. It scored 41.4% ± 1.6%. This is different from the shuffled control: the control was trained on shuffled sequences; the 41.4% check shuffled the input to the already-trained ordered model.

The five seeds change training randomness, not the player split. The confusion matrices show seed 42 results; they are not averages across the five seeds.

---

## 5. Main limitations

1. The test set contains eight players from one fixed split.
2. The videos are staged indoor clips, not real matches.
3. The ball is not visible, and the models do not use useful ball-trajectory or racquet-angle information.
4. Pose detection can make mistakes, even after quality checks and interpolation.
5. The image and skeleton pipelines use different inputs, so their scores are not a fully matched comparison.

---

## 6. References

1. Gourgari et al. [THETIS: Three Dimensional Tennis Shots—A Human Action Dataset](https://openaccess.thecvf.com/content_cvpr_workshops_2013/W08/papers/Gourgari_THETIS_Three_Dimensional_2013_CVPR_paper.pdf). CVPR Workshops, 2013.
2. He et al. [Deep Residual Learning for Image Recognition](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html). CVPR, 2016.
3. Russakovsky et al. [ImageNet Large Scale Visual Recognition Challenge](https://doi.org/10.1007/s11263-015-0816-y). International Journal of Computer Vision, 2015.
4. Hovad et al. [Classification of Tennis Actions Using Deep Learning](https://arxiv.org/abs/2402.02545). arXiv, 2024.
5. Feichtenhofer et al. [SlowFast Networks for Video Recognition](https://openaccess.thecvf.com/content_ICCV_2019/html/Feichtenhofer_SlowFast_Networks_for_Video_Recognition_ICCV_2019_paper.html). ICCV, 2019.
6. Jocher and Qiu. [Ultralytics YOLO11](https://docs.ultralytics.com/models/yolo11). Ultralytics documentation, 2024.
7. Lin et al. [Microsoft COCO: Common Objects in Context](https://doi.org/10.1007/978-3-319-10602-1_48). ECCV, 2014.
8. Kingma and Ba. [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980). ICLR, 2015.
