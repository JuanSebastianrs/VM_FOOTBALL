# VM_FOOTBALL: A Comprehensive Computer Vision System for Tactical Football Analysis

**Abstract**  
Tactical analysis in modern football is increasingly driven by automated spatio-temporal data extraction from match videos. This paper presents VM_FOOTBALL, an end-to-end computer vision system designed to track players, identify the ball, and classify teams dynamically. By integrating state-of-the-art deep learning architectures—specifically YOLO and RF-DETR for object detection, coupled with ByteTrack for multi-object tracking—the system addresses inherent challenges such as severe occlusions, motion blur, and the diminutive scale of the ball. Furthermore, the project introduces a highly scalable, cost-effective training infrastructure leveraging Google Cloud Platform's Vertex AI Spot Jobs. This document details the system's operational methodologies, the main results achieved, and future directions for advanced tactical analytics.

---

## 1. Introduction

The evolution of football analytics has shifted from manual event tagging to continuous, automated tracking of player and ball coordinates. This spatio-temporal data provides coaches and analysts with critical insights into team formations, pressing triggers, and spatial dominance. However, developing a robust automated tracking system from standard broadcast footage presents significant computer vision challenges. Players frequently occlude one another during set pieces or tackles, cameras pan and zoom rapidly causing motion blur, and the ball often occupies only a fraction of a percent of the total image area.

The **VM_FOOTBALL** project was conceived as a comprehensive thesis to address these challenges, with a design philosophy emphasizing adaptability, state-of-the-art accuracy, and infrastructural scalability. Unlike proprietary systems that rely on multi-camera setups installed in elite stadiums, this system is designed to operate on single-view broadcast footage, making advanced tactical analysis accessible to a broader range of football organizations, particularly within the Latin American context.

The primary objective of the system is to construct a continuous pipeline that ingests raw video, detects key entities (players, goalkeepers, referees, the ball), tracks their identities over time, and dynamically clusters them into distinct teams without prior manual annotation.

---

## 2. Methodology and System Architecture

The VM_FOOTBALL system is engineered as a multi-stage pipeline, refining raw visual data into structured analytical outputs. The architecture is divided into three primary modules: Object Detection, Multi-Object Tracking, and Identity/Team Classification, supported by a scalable cloud training backbone.

### 2.1. Advanced Object Detection Strategy

Accurate object detection serves as the foundation of the pipeline. To isolate the fast-moving football and accurately bound players in dense crowds, the system employs a hybrid detection strategy:

- **YOLO Pipeline (Ultralytics):** The YOLO architectural family is utilized for its exceptional balance of real-time inference speed and accuracy. It is particularly effective for isolating the ball, which requires high-frequency bounding box updates to maintain trajectory continuity.
- **RF-DETR (Real-Time DEtection TRansformer):** To overcome the limitations of traditional CNN-based detectors in dense scenes, the system integrates RF-DETR for player and goalkeeper detection. Unlike traditional methods that rely on anchor boxes and Non-Maximum Suppression (NMS)—which often inaccurately filter out overlapping players during close-contact plays like corner kicks—RF-DETR utilizes a transformer-based encoder-decoder architecture. By framing detection as a set prediction problem solved via bipartite matching, it excels in crowded scenarios and generalizes better to varying camera angles.

### 2.2. Robust Multi-Object Tracking

Once entities are detected in individual frames, the system must link these detections chronologically to form cohesive trajectories.
- **ByteTrack Integration:** The system employs the ByteTrack algorithm for multi-object tracking. Traditional trackers often discard low-confidence detections, leading to fragmented trajectories when a player is partially occluded. ByteTrack innovates by associating almost every detection box; it prioritizes high-confidence boxes but actively utilizes low-confidence boxes to bridge gaps during mutual occlusion, resulting in highly continuous player tracking even in complex broadcast angles.

### 2.3. Unsupervised Team Classification

Differentiating teams dynamically is crucial for semantic tactical analysis.
- **K-Means Clustering on HSV Color Space:** The system utilizes an unsupervised clustering approach to separate players into opposing teams. By extracting the bounding box for each tracked player and converting the pixel data into the HSV (Hue, Saturation, Value) color space, the system mitigates the effects of uneven stadium lighting and shadow casting. A K-Means clustering algorithm is then applied to the dominant color features of the players' jerseys, effectively grouping them into their respective teams without relying on pre-trained team-specific models.

### 2.4. Cloud-Native Model Training Infrastructure

Training heavy transformer models like RF-DETR on massive datasets (such as SoccerNet-MOT) requires substantial computational resources. A cornerstone of the VM_FOOTBALL project is its enterprise-grade, cloud-native training infrastructure.
- **Scalable Vertex AI Orchestration:** The system leverages Google Cloud Platform's Vertex AI to execute custom training jobs. By utilizing Preemptible (Spot) GPU instances (such as NVIDIA T4 and L4), the project drastically reduces compute costs by up to 90% compared to on-demand pricing. 
- **Resilience and Automated Data Engineering:** The pipeline is engineered for supreme fault tolerance. It features background threads that automatically synchronize training checkpoints (e.g., neural network weights, optimizer states) to Google Cloud Storage (GCS) at regular intervals. It also handles the automated, on-the-fly conversion of standard YOLO label formats into complex COCO JSON schemas required by the RF-DETR architecture, streamlining the data preparation layer.

---

## 3. Main Results

The VM_FOOTBALL system has achieved several critical milestones, validating both its theoretical approach and its software engineering architecture:

1. **High-Fidelity Tracking in Broadcast Video:**
   The integration of YOLO/RF-DETR with ByteTrack has yielded a highly functional inference pipeline capable of processing standard broadcast clips. The system successfully maintains player identities through brief occlusions and complex camera movements, providing a reliable foundation for trajectory analysis.

2. **Effective Unsupervised Team Separation:**
   The applied K-Means and HSV color space methodology has proven highly effective. Upon processing validation videos, the system accurately delineates the two opposing teams and distinguishes goalkeepers/referees, adapting on-the-fly to varying jersey color combinations without manual dataset annotation.

3. **Cost-Effective, Enterprise-Scale Training Capability:**
   The strategic deployment of Vertex AI Spot Jobs has transformed the model training paradigm of the project. The infrastructure successfully handles the rigorous demands of training transformer-based detection models (e.g., batches of 32 effective, 480px resolution) while autonomously recovering from instance preemptions via its robust GCS checkpointing system.

4. **Streamlined Data Pipelines:**
   The system successfully orchestrated the ingestion and transformation of the expansive SoccerNet-MOT dataset. By automating bulk downloads and implementing real-time label conversion mechanics, the data preparation overhead was minimized, allowing rapid iteration on hyperparameter tuning.

---

## 4. Conclusion and Future Directions

The VM_FOOTBALL project represents a robust, well-architected computer vision pipeline capable of bridging the gap between raw sports broadcast footage and structured tactical datasets. By harmonizing edge-efficient paradigms with state-of-the-art vision transformers and grounding the development in resilient, cloud-native infrastructure, the project establishes a firm baseline for democratized tactical analysis.

**Future development vectors include:**
- **Zero-Shot Segmentation (SAM2):** Integrating the Segment Anything Model 2 to transition from rigid bounding boxes to pixel-perfect player segmentation masks, enabling advanced pose and body orientation analytics.
- **Advanced Player Identification:** Replacing basic color clustering with sophisticated visual embeddings (SigLIP2) and incorporating OCR models (SmolVLM2, ResNet) to automatically read jersey numbers for concrete, roster-linked player identification.
- **Automated Event Detection:** Leveraging the structured spatio-temporal coordinate data to automatically identify key match events, such as passes, shots, and offside line evaluations.

---

## 5. References

1. **Jocher, G., et al. (2024).** *Ultralytics YOLO (v11)*. Ultralytics. [https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)
2. **Zhao, Y., et al. (2023).** *DETRs Beat YOLOs on Real-time Object Detection*. arXiv preprint arXiv:2304.08069.
3. **Zhang, Y., et al. (2022).** *ByteTrack: Multi-Object Tracking by Associating Every Detection Box*. European Conference on Computer Vision (ECCV).
4. **Cioppa, A., et al. (2022).** *SoccerNet-Tracking: Multiple Object Tracking Dataset and Benchmark in Soccer Videos*. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops.
5. **Ravi, N., et al. (2024).** *SAM 2: Segment Anything in Images and Videos*. Meta AI Research.
6. **Zhai, X., et al. (2023).** *Sigmoid Loss for Language Image Pre-Training (SigLIP)*. International Conference on Computer Vision (ICCV).
