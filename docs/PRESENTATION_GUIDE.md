# Presentation guide

The reviewed deck is `presentation/final/TrustRAG_Project_Review.pptx`. Allow about 12 minutes plus a short demonstration. Speaker notes include the evidence and sources behind each slide.

Open with the unsupported Yarrow answer, then explain how uncertainty and evidence guide the pipeline. Distinguish the frozen language model from the small trained detector. Explain the train/validation/test split before discussing scores.

On the comparison slides, state the narrow task and the historical timing convention. The results support retrieval on these questions. Adaptive routing added overhead, and no intervention triggered. Avoid claiming a correction advantage.

For the demonstration, start model mode with `python run.py --model`. Show a question, its retrieved passage, the result trace, and JSON export. Explain that document lookup is an extractive mode and that adaptive acceptance can still be unverified.

When discussing limitations, emphasize the small fictional dataset, imperfect NLI judgments, missed contradictions, and the need for broader evaluation. Close with the repository and presentation links.
