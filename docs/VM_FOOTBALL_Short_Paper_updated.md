%%%%%%%%%%%%%%%%%%%% author.tex %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% sample root file for your "contribution" to a contributed volume
%
% Use this file as a template for your own input.
%
%%%%%%%%%%%%%%%% Springer %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% RECOMMENDED %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\documentclass[graybox]{svmult}

% choose options for [] as required from the list
% in the Reference Guide

\usepackage{type1cm}        % activate if the above 3 fonts are
                            % not available on your system
%
\usepackage{makeidx}         % allows index generation
\usepackage{graphicx}        % standard LaTeX graphics tool
                             % when including figure files
\usepackage{multicol}        % used for the two-column index
\usepackage[bottom]{footmisc}% places footnotes at page bottom


\usepackage{newtxtext}       % 
\usepackage{newtxmath}       % selects Times Roman as basic font

% see the list of further useful packages
% in the Reference Guide

\makeindex             % used for the subject index
                       % please use the style svind.ist with
                       % your makeindex program

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\begin{document}

\title*{Tactical Vision: A Comprehensive Computer Vision System for Tactical Football Analysis}
% Use \titlerunning{Short Title} for an abbreviated version of
% your contribution title if the original one is too long
\author{Juan Sebastian Rodriguez and Alejandro Rafael Vega}
% Use \authorrunning{Short Title} for an abbreviated version of
% your contribution title if the original one is too long
\institute{Juan Sebastian Rodriguez and \at Universidad del Rosario, Bogotá Colombia, \email{Juansebasti.rodrig10@urosario.edu.co}
\and Alejandro Rafael Vegar \at Universidad del Rosario, Bogotá Colombia\email{alejandror.vega@urosario.edu.co}}
%
% Use the package "url.sty" to avoid
% problems with special characters
% used in your e-mail or web address
%
\maketitle

\abstract*{Tactical analysis in modern football is increasingly driven by automated spatio-temporal data extraction from match videos. This paper presents Tactical Vision, an end-to-end computer vision system designed to track players, identify the ball, and classify teams dynamically. By integrating state-of-the-art deep learning architectures—specifically YOLO and RF-DETR for object detection, coupled with ByteTrack for multi-object tracking—the system addresses inherent challenges such as severe occlusions, motion blur, and the diminutive scale of the ball. Furthermore, the project introduces a highly scalable, cost-effective training infrastructure leveraging Google Cloud Platform's Vertex AI Spot Jobs. This document details the system's operational methodologies, the main results achieved, and future directions for advanced tactical analytics.}

\abstract{Tactical analysis in modern football is increasingly driven by automated spatio-temporal data extraction from match videos. This paper presents Tactical Vision, an end-to-end computer vision system designed to track players, identify the ball, and classify teams dynamically. By integrating state-of-the-art deep learning architectures—specifically YOLO and RF-DETR for object detection, coupled with ByteTrack for multi-object tracking—the system addresses inherent challenges such as severe occlusions, motion blur, and the diminutive scale of the ball. Furthermore, the project introduces a highly scalable, cost-effective training infrastructure leveraging Google Cloud Platform's Vertex AI Spot Jobs. This document details the system's operational methodologies, the main results achieved, and future directions for advanced tactical analytics.}

\section{Introduction}
\label{sec:1}

The evolution of football analytics has shifted from manual event tagging to continuous, automated tracking of player and ball coordinates. This spatio-temporal data provides coaches and analysts with critical insights into team formations, pressing triggers, and spatial dominance. However, developing a robust automated tracking system from standard broadcast footage presents significant computer vision challenges. Players frequently occlude one another during set pieces or tackles, cameras pan and zoom rapidly causing motion blur, and the ball often occupies only a fraction of a percent of the total image area.

The \textbf{Tactical Vision} project was conceived as a comprehensive master's thesis to address these challenges, with a design philosophy emphasizing adaptability, state-of-the-art accuracy, and infrastructural scalability. Unlike proprietary systems that rely on multi-camera setups installed in elite stadiums, this system is designed to operate on single-view broadcast footage, making advanced tactical analysis accessible to a broader range of football organizations, particularly within the Latin American context.

The primary objective of the system is to construct a continuous pipeline that ingests raw video, detects key entities (players, goalkeepers, referees, the ball), tracks their identities over time, and dynamically clusters them into distinct teams without prior manual annotation.

\section{Methodology and System Architecture}
\label{sec:2}

The Tactical Vision system is engineered as a multi-stage pipeline, refining raw visual data into structured analytical outputs.

\subsection{Advanced Object Detection Strategy}
\label{subsec:2_1}

Accurate object detection serves as the foundation of the pipeline. To isolate the fast-moving football and accurately bound players in dense crowds, the system employs a hybrid detection strategy:

\begin{itemize}
\item \textbf{YOLO Pipeline (Ultralytics):} The localization of the ball constitutes an extreme challenge due to it being a small object in match recordings and its high velocity. For this task, the \textbf{YOLOv11n} architecture was implemented, prioritizing computational efficiency. Although recent transformer-based models (such as RF-DETRv2) achieve state-of-the-art accuracy on generic datasets, recent literature demonstrates that YOLO11 maintains a successful detection rate ($>$90\%) with drastically lower latency in video inference (Wang, 2025; Akbarnezhad, 2025). 

To evaluate the design, an experiment measuring the \textbf{mAP50} exclusive to the ball class was conducted over 35 epochs (see Table~\ref{tab:1}). The initial model (\textit{Exp\_0\_Small\_640\_Base}) reached a peak mAP50 of only 0.398. By scaling the input resolution to \textbf{1280 pixels} (\textit{1280\_Baseline}) to prevent the ball from vanishing in convolutional layers, performance increased to 0.521. Finally, the definitive model (\textit{1280\_DataCentric}) stabilized the curve with a mAP50 of 0.509 using a strict data-centric strategy: the \textbf{Mixup} algorithm was prohibited, as its transparency interpolation destroys the ball's scarce pixels. Instead, \textbf{Copy-Paste} synthetic injection was prioritized to counteract class imbalance, a decision supported by 2025 empirical studies in football analytics (Puspita et al., 2025; Wu et al., 2025).
\end{itemize}

\begin{quotation}
\textbf{Caption:} Experimental results for ball localization. \textbf{mAP50} (Mean Average Precision) measures the accuracy of the bounding box at a 50\% IoU threshold; it quantifies the overlap between the predicted box and the ground truth.
\end{quotation}

\begin{itemize}
\item \textbf{RF-DETR (Real-Time DEtection TRansformer):} The finalized player detector uses RFDETRBase (31.8M parameters) trained as a single-class model (player) at 448px for 60 epochs with effective batch size 32. During training, SoccerNet classes player\_left, player\_right, goalkeeper\_left, and goalkeeper\_right are mapped into player, while referee and ball are excluded from the objective. This design prioritizes robust player recall in crowded broadcast scenes and delegates semantic differentiation to downstream modules.
\end{itemize}

\subsection{Robust Multi-Object Tracking}
\label{subsec:2_2}

For the temporal association of detected objects (MOT), the system discards classic algorithms such as SORT and DeepSORT in favor of the \textbf{ByteTrack} architecture. In the context of televised soccer, the high density of players, mutual occlusions, and motion blur cause transient drops in the confidence scores of the spatial detector's predictions. DeepSORT, by heuristic design, summarily eliminates any bounding box that does not exceed a rigid confidence threshold, resulting in the destructive fragmentation of trajectories and an unacceptable increase in Identity Switches ($IDSW$). Furthermore, DeepSORT's reliance on visual appearance convolutional networks (Re-ID) proves inefficient in soccer, where the ball lacks discriminable texture and players from the same team share nearly identical chromatic characteristics.

ByteTrack mitigates this mathematical problem by implementing a structured bipartite association. In its secondary phase, the algorithm actively rescues low-confidence predictions (previously discarded) and associates them via spatial overlap ($IoU$) with trajectories projected by the Kalman Filter. In standardized evaluations on the soccer-specific \textbf{SoccerNet-Tracking} dataset, ByteTrack has empirically demonstrated its superiority over DeepSORT, increasing the comprehensive $HOTA$ (Higher Order Tracking Accuracy) metric from 69.6\% to 71.5\%, and substantially improving Association Accuracy ($AssA$) from 58.7\% to 60.7\% in the face of continuous occlusions.

\subsection{Unsupervised Team Classification}
\label{subsec:2_3}

Differentiating teams dynamically is crucial for semantic tactical analysis.
\begin{itemize}
\item \textbf{K-Means Clustering on HSV Color Space:} Team assignment is addressed with a three-mode formulation (HSV, SigLIP, and hybrid), where the current baseline is HSV-driven unsupervised clustering. Each tracked player is represented with a torso crop (top=0.15, bottom=0.60, left=0.20, right=0.80), converted to HSV, and filtered with a pitch-green suppression mask ($H \in [35, 90]$) to reduce grass contamination. A normalized H/S histogram descriptor (12 bins per channel) is aggregated at track level and clustered with K-Means ($k=2$), followed by a left/right heuristic on early frames to stabilize cluster identity across time.
\end{itemize}

\subsection{Cloud-Native Model Training Infrastructure}
\label{subsec:2_4}

Training heavy transformer models like RF-DETR on massive datasets (such as SoccerNet-MOT) requires substantial computational resources. A cornerstone of the Tactical Vision project is its enterprise-grade, cloud-native training infrastructure.
\begin{itemize}
\item \textbf{Scalable Vertex AI Orchestration:} The system leverages Google Cloud Platform's Vertex AI to execute custom training jobs. By utilizing Preemptible (Spot) GPU instances (such as NVIDIA T4 and L4), the project drastically reduces compute costs by up to 90\% compared to on-demand pricing. 
\item \textbf{Resilience and Automated Data Engineering:} The pipeline is engineered for supreme fault tolerance. It features background threads that automatically synchronize training checkpoints (e.g., neural network weights, optimizer states) to Google Cloud Storage (GCS) at regular intervals. It also handles the automated, on-the-fly conversion of standard YOLO label formats into complex COCO JSON schemas required by the RF-DETR architecture, streamlining the data preparation layer.
\end{itemize}

\section{Main Results}
\label{sec:3}

\label{subsec:3_1}
\subsubsection{Quantitative YOLO Training Results}

\begin{table}[htbp]
\caption{Ball Localization Performance}
\label{tab:1}       % Give a unique label
\begin{tabular}{p{3.5cm}p{2cm}p{2.5cm}p{2.5cm}}
\hline\noalign{\smallskip}
Model Version & Resolution & Augmentation & mAP50 (Ball)  \\
\noalign{\smallskip}\svhline\noalign{\smallskip}
Exp\_0\_Small\_640\_Base & 640 px & Standard & 0.398\\
1280\_Baseline & 1280 px & Standard & 0.521\\
1280\_DataCentric & 1280 px & Copy-Paste & 0.509\\
\noalign{\smallskip}\hline\noalign{\smallskip}
\end{tabular}
\end{table}

\subsubsection{Quantitative RF-DETR Training Results}

\begin{table}[htbp]
\caption{The final RF-DETR training configuration yields the following metrics:}
\label{tab:2}       % Give a unique label
\begin{tabular}{p{3.5cm}p{1.5cm}p{1.8cm}p{1.5cm}p{1.5cm}p{1.2cm}}
\hline\noalign{\smallskip}
Split & mAP@50 & mAP@50:95 & Precision & Recall & F1 \\
\noalign{\smallskip}\svhline\noalign{\smallskip}
Validation (best EMA) & 95.85\% & 58.27\% & 95.93\% & 94.78\% & 95.35\% \\
Test & 95.41\% & 56.97\% & 96.03\% & 94.24\% & 95.13\% \\
\noalign{\smallskip}\hline\noalign{\smallskip}
\end{tabular}
\end{table}

\subsection{RF-DETR Visual Inference Summary}
\label{subsec:3_2}

On the SNMOT-116 sequence (750 frames), the system processed the full video with an average of \textbf{12.7 detections per frame} and \textbf{64.6 ms per frame} (approximately \textbf{15.5 FPS} on local GPU).

\subsection{Clustering Outputs (Current Baseline)}
\label{subsec:3_3}

The HSV clustering baseline on SNMOT-116 produced \textbf{84} track assignments, with cluster distribution \textbf{39/45} (team 0/team 1) and \textbf{8,690} accumulated feature samples. This confirms operational viability of the current color-based baseline for team separation in broadcast footage.

\section{Conclusion and Future Directions}
\label{sec:4}

The Tactical Vision project represents a robust, well-architected computer vision pipeline capable of bridging the gap between raw sports broadcast footage and structured tactical datasets. By harmonizing edge-efficient paradigms with state-of-the-art vision transformers and grounding the development in resilient, cloud-native infrastructure, the project establishes a firm baseline for democratized tactical analysis.

\textbf{Future development vectors include:}
\begin{itemize}
\item \textbf{Zero-Shot Segmentation (SAM2):} Integrating the Segment Anything Model 2 to transition from rigid bounding boxes to pixel-perfect player segmentation masks, enabling advanced pose and body orientation analytics.
\item \textbf{Advanced Player Identification:} Replacing basic color clustering with sophisticated visual embeddings (SigLIP2) and incorporating OCR models (SmolVLM2, ResNet) to automatically read jersey numbers for concrete, roster-linked player identification.
\item \textbf{Automated Event Detection:} Leveraging the structured spatio-temporal coordinate data to automatically identify key match events, such as passes, shots, and offside line evaluations.
\end{itemize}
\input{references}
\end{document}