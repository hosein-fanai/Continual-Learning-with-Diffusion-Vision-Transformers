# Notebook import initialization

`import init` makes repository modules importable and sets the working directory
to the repository root resolved from the helper's file location. It works from
the root and the notebook directory without a machine-specific checkout path.

Importing the helper changes process-wide import and working-directory state.
The retained runtime initialization call does not impose a GPU memory cap.
Use a fresh TensorFlow 2.20 kernel and configure device memory before the first
GPU operation when custom limits are required.
