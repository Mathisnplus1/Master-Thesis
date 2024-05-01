# Master-Thesis
Artificial neurogenesis in the context of hardware-constrained continual learning

Catastrophic forgetting is a major challenge in continual learning systems, where a neural network needs to adapt to new tasks that differ from what it has previously learned. To tackle this, experts have devised strategies to either keep the neural network's critical weights intact or use generative models to refresh the network's memory of past tasks. However, these methods aren't perfect for hardware implementation due to their additional memory demands or the need for a separate network that constantly rehearses past information.



Meanwhile, we understand that a system's architecture significantly impacts its ability to generalize knowledge. Enter Gradmax [1], an innovative algorithm that simultaneously optimizes a network's structure and weights, and smartly allocates neurons for new tasks without disrupting existing knowledge. This method is memory and computation efficient, making it suitable for hardware applications. But it does require hardware capable of dynamically adjusting its architecture to integrate new neurons as needed. Our recent work has led to a flexible hardware architecture, Mosaic [2], with multiple cores that can rearrange their connections in real-time.

 

The aim of this project is to explore the integration of Gradmax with our dynamic 'Mosaic' architecture. We plan to approach this by factoring hardware limitations into the optimization process. In addition, we're considering the development of a gating network within this architecture that determines the optimal timing for assigning tasks to newly added neurons. This could leverage cutting-edge techniques like Mixture of Expert models [3].

 

By refining the synergy between Gradmax and the Mosaic architecture, we hope to pave the way for more resilient and adaptable hardware-based continual learning systems.

 

 

[1] Evici, et al, ICLR 2022, “GradMax: Growing Neural Networks using Gradient Information”

[2] Dalgaty, Moro, Demirag, et al, Nature communications 2024, “Mosaic: in-memory computing and routing for small-world spike-based neuromorphic systems”

[3] Eigne, et al, “Learning Factored Representations in a Deep Mixture of Experts”
