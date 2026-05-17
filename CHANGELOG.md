# Changelog

## Unreleased

- Fixed dataset image loading to feed RGB images to backbones. Previous OpenCV-backed loading passed BGR channel order through the training and evaluation pipeline, so metrics from newly trained RGB models are not directly comparable with legacy BGR-trained checkpoints.
