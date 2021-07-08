# Uncertainty-Vision
Use of Metropolis algorithm (from Scalismo library) and image processing (from BoofCV library) in order to experiment with the camera uncertainty calculations of Collaborative Robots and improve the accuracy through statistical analysis.
This code is using an OOP (object-oriented programming) language
Scala version: 2.12.8

# Main Libraries
BoofCV [https://github.com/lessthanoptimal/BoofCV/tree/v0.34]

Scalismo [https://github.com/unibas-gravis/scalismo]

# Object: UncertaintyVision
This object includes 5 classes:
  - Proposal Generator for Translation coordinates: CameraTProposal
  - Proposal Generator for Rotation coordinates: CameraRProposal
  - Proposal Generator for intrinsic variable Focal length: IntrinsicProposal
  - Uncertainty of the detected fiducials measurements: ImageFiducialLikelihood
  - Logger to return accepted and rejected proposals: ProposalLogger

This object incluse 4 methods:
  - To project uncertainties in the world to the image plane: projectWorldUncertaintyToImage
  - Get average and Std Deviation of spherical coordinates: SphericalGeomStats
  - Get average and Std Deviation of euclidean coordinates: EuclideanGeomStats
  - Get camera with the focal length parameters from the new proposal: PoseWithFocalLengthToCamera
