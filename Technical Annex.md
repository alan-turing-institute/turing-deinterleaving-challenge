# Technical Annex: Radar Deinterleaving and Pulse Descriptor Words

## 1. Introduction

Radar deinterleaving is a critical signal processing task in electronic warfare, surveillance, and radar signal intelligence applications. When multiple radar emitters operate simultaneously within the same electromagnetic environment, their transmitted pulses become interleaved in time, creating complex pulse trains that must be separated and attributed to their originating sources.

## 2. Radar Pulse Deinterleaving Problem

### 2.1 Problem Definition

![PDW Schema](.assets/Ta/Schema.png)

*Figure 1: Schematic representation of deinterleaving protocol*

The radar pulse deinterleaving problem involves separating radar pulses from multiple unknown emitters present in a single recorded pulse train. This separation task is particularly challenging because:

- The number of active emitters is typically unknown a priori
- Pulse patterns may be irregular or adaptive
- Environmental factors introduce noise and measurement uncertainty
- Real-time processing constraints limit computational complexity

### 2.2 Mathematical Formulation

Let $ \vec{X} = \lbrace x_{1}, x_{2}, \dots,x_{n} \rbrace $ represent a pulse train containing n pulses from N unknown emitters. The deinterleaving task seeks to partition $vec{X}$ into N disjoint subsets:

### $$ \vec{X} = \lbrace U_{1},\dots U_{N} \rbrace $$

where each subset $U_{i}$ contains all pulses originating from emitter i.

## 3. Pulse Descriptor Words (PDWs)

### 3.1 PDW Definition

A Pulse Descriptor Word (PDW) is a multi-dimensional feature vector that characterises the measurable parameters of a radar pulse. PDWs serve as the fundamental input for deinterleaving algorithms, providing quantitative descriptions of pulse characteristics.

### 3.2 Standard PDW Parameters

The most commonly used PDW parameters include:

#### 3.2.1 Time of Arrival (ToA)
- **Definition**: Timestamp when the pulse leading edge is detected
- **Units**: Microseconds $\mu s$
- **Significance**: Enables temporal pattern analysis and pulse repetition interval (PRI) estimation

#### 3.2.2 Centre Frequency (CF)
- **Definition**: Carrier frequency of the radar pulse
- **Units**: Megahertz (MHz) or Gigahertz (GHz)
- **Significance**: Primary discriminator for frequency-agile or fixed-frequency emitters

#### 3.2.3 Pulse Width (PW)
- **Definition**: Duration of the pulse envelope
- **Units**: Microseconds ($\mu s$)
- **Significance**: Indicates radar type and operational mode

#### 3.2.4 Angle of Arrival (AoA)
- **Definition**: Spatial direction from which the pulse arrives
- **Units**: Degrees (�) or radians
- **Significance**: Provides spatial discrimination between emitters

#### 3.2.5 Amplitude/Power
- **Definition**: Peak or integrated power level of the received pulse
- **Units**: Decibels (dB) or linear scale
- **Significance**: Relates to emitter power and propagation distance

### 3.3 Extended PDW Features

Advanced deinterleaving systems may incorporate additional features extracted from in-phase and quadrature (IQ) data:

- **Modulation type**: Linear frequency modulation (LFM), phase-shift keying (PSK), etc.
- **Bandwidth**: Spectral width of the pulse
- **Rise/fall times**: Pulse envelope characteristics
- **Intra-pulse features**: Spectral content, phase variations

## 4. Traditional Deinterleaving Approaches

### 4.1 Histogram-Based Methods

Classical approaches analyse statistical distributions of PDW parameters:

- **PRI histograms**: Identify repetitive timing patterns
- **Frequency clustering**: Group pulses by carrier frequency
- **Joint parameter analysis**: Multi-dimensional histogram techniques

### 4.2 Sequence-Based Methods

These methods exploit temporal ordering and pattern recognition:

- **PRI sequence matching**: Detect repeating timing sequences
- **Markov models**: Model state transitions in pulse patterns
- **Autocorrelation analysis**: Identify periodic components

### 4.3 Clustering Techniques

Unsupervised learning approaches partition pulse space:

- **K-means clustering**: Assumes spherical clusters in feature space
- **DBSCAN**: Density-based clustering for non-spherical distributions
- **Hierarchical clustering**: Tree-based partitioning methods

## 5. Modern Deep Learning Approaches

### 5.1 Transformer-Based Metric Learning

Recent advances leverage transformer architectures for deinterleaving:

#### 5.1.1 Architecture
- **Sequence-to-sequence models**: Process entire pulse trains simultaneously
- **Self-attention mechanisms**: Capture long-range dependencies between pulses
- **Embedding generation**: Transform PDWs into discriminative feature representations

#### 5.1.2 Training Methodology
- **Triplet loss function**: Optimises embedding similarity within emitters and dissimilarity between emitters
- **Synthetic data generation**: Creates controlled training scenarios with known ground truth
- **Metric learning objective**: Learns distance functions rather than direct classification

### 5.2 Performance Metrics

Deinterleaving performance is evaluated using clustering metrics:

- **Adjusted Mutual Information (AMI)**: Measures clustering quality adjusted for chance
- **Silhouette coefficient**: Quantifies cluster separation and cohesion
- **Homogeneity and completeness**: Evaluate cluster purity and coverage

## 6. Data Preprocessing and Normalisation

### 6.1 PDW Normalisation Strategies

Effective deinterleaving requires careful preprocessing:

#### 6.1.1 Time of Arrival
- **Linear rescaling**: Map ToA values to [0,1] interval
- **Relative timing**: Compute inter-pulse intervals

#### 6.1.2 Frequency and Amplitude
- **Statistical normalisation**: Apply z-score normalisation
- **Robust scaling**: Use median and interquartile range

#### 6.1.3 Angular Parameters
- **Circular normalisation**: Handle angular wraparound effects
- **Directional statistics**: Apply appropriate circular measures

## 7. Challenges and Limitations

### 7.1 Technical Challenges

- **Missing pulses**: Incomplete pulse reception due to propagation effects
- **Measurement noise**: PDW parameter uncertainty
- **Overlapping parameters**: Emitters with similar characteristics
- **Adaptive waveforms**: Time-varying pulse patterns

### 7.2 Operational Constraints

- **Real-time processing**: Latency requirements for operational systems
- **Computational complexity**: Resource limitations in deployed systems
- **Scalability**: Performance with increasing numbers of emitters

## 8. Applications and Future Directions

### 8.1 Military Applications
- Electronic warfare systems
- Radar threat warning systems
- Signals intelligence platforms

### 8.2 Civilian Applications
- Air traffic control systems
- Spectrum monitoring and management
- Interference mitigation in radar networks

### 8.3 Research Directions
- **Multi-modal fusion**: Incorporating additional sensor modalities
- **Online learning**: Adaptive algorithms for dynamic environments
- **Explainable AI**: Interpretable deinterleaving decisions
- **Federated learning**: Distributed training across multiple platforms

## References

Based on: "Radar Pulse Deinterleaving with Transformer Based Deep Metric Learning" (arXiv:2503.13476), which presents a novel transformer-based approach achieving an adjusted mutual information score of 0.882 on synthetic radar pulse data using 5-dimensional PDWs (ToA, centre frequency, pulse width, AoA, and amplitude).