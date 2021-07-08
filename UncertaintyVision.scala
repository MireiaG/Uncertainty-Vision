package r3bot

import boofcv.alg.distort.brown.LensDistortionBrown
import boofcv.alg.geo.PerspectiveOps
import boofcv.struct.calib.{CameraModel, CameraPinholeBrown}
import georegression.struct.point.{Point2D_F64, Point3D_F64}
import r3bot.CameraInTheWorld.{AverageEstimation, RobustEstimation}
import r3bot.QRVision1.{cameraModel, logger, streamCamera}
import scalismo.registration.LandmarkRegistration
import scalismo.sampling._
import scalismo.sampling.proposals._
import scalismo.sampling.evaluators._
import scalismo.sampling.algorithms._
import scalismo.sampling.loggers.AcceptRejectLogger
import scalismo.sampling.proposals.MixtureProposal.implicits._
import scalismo.utils.Random

import scala.collection.{immutable, mutable}

object UncertaintyVision {

  implicit val random: Random = Random(56L)

  case class CameraTProposal(proposalTSdev: Double)(implicit rnd: Random)
    extends ProposalGenerator[PoseWithFocalLength] with SymmetricTransitionRatio[PoseWithFocalLength] {
    override def propose(current: PoseWithFocalLength): PoseWithFocalLength = {
      // given `current` -> create a new camera proposal
      val x = current.location.x + proposalTSdev * rnd.scalaRandom.nextGaussian()
      val y = current.location.y + proposalTSdev * rnd.scalaRandom.nextGaussian()
      val z = current.location.z + proposalTSdev * rnd.scalaRandom.nextGaussian()
      PoseWithFocalLength(Location(x, y, z), current.orientation, current.focallengthX, current.focallengthY)
    }
  }
  case class CameraRProposal(proposalRSdev: Double)(implicit rnd: Random)
    extends ProposalGenerator[PoseWithFocalLength] with SymmetricTransitionRatio[PoseWithFocalLength] {
    override def propose(current: PoseWithFocalLength): PoseWithFocalLength = {
      val xAngle = current.orientation.xAngle + proposalRSdev * rnd.scalaRandom.nextGaussian()
      val yAngle = current.orientation.yAngle + proposalRSdev * rnd.scalaRandom.nextGaussian()
      val zAngle = current.orientation.zAngle + proposalRSdev * rnd.scalaRandom.nextGaussian()
      // new angles are range limited, so that angles > than 360 deg reshape to the deg from 0
      PoseWithFocalLength(
        current.location,
        Orientation(xAngle % (2 * math.Pi), yAngle % (2 * math.Pi), zAngle % (2 * math.Pi)),
        current.focallengthX, current.focallengthY
      )
    }
  }
  case class IntrinsicProposal(focallengthProposalSdev: Double)(implicit rnd: Random)
    extends ProposalGenerator[PoseWithFocalLength] with SymmetricTransitionRatio[PoseWithFocalLength] {
    override def propose(current: PoseWithFocalLength): PoseWithFocalLength = {
      val focallengthX = current.focallengthX + focallengthProposalSdev * rnd.scalaRandom.nextGaussian()
      val focallengthY = current.focallengthY + focallengthProposalSdev * rnd.scalaRandom.nextGaussian()
      PoseWithFocalLength(current.location, current.orientation, focallengthX, focallengthY)
    }
  }

  /// project uncertainties in the world to the image plane
  // using the specified world-to-image transform and evaluate around the anchor point
  def projectWorldUncertaintyToImage(dx: Double, dy: Double, dz: Double,
                                     worldToImage: Point3D_F64 => Point2D_F64,
                                     anchorPoint: Point3D_F64): (Double, Double) = {
    // idea:
    // build 8 points of 3d cube dx, dy, dz around anchorPoint
    // project all to image plane
    // take min/max in x and y direction
    val cubePoints: IndexedSeq[(Double, Double, Double)] = IndexedSeq(
      (0,   0,  0),
      (dx,  0,  0),
      (0,  dy,  0),
      (0,   0, dz),
      (dx, dy,  0),
      (0,  dy, dz),
      (dx,  0, dz),
      (dx, dy, dz)
    )
    // place around anchor point
    val anchoredCube = cubePoints.map{case (x, y, z) =>
      new Point3D_F64(
        anchorPoint.x + x - dx/2,
        anchorPoint.y + y - dy/2,
        anchorPoint.z + z - dz/2)
    }
    // project to image
    val imagePoints = anchoredCube.map(worldToImage)
    // min / max
    val minY = imagePoints.map(_.y).min
    val maxY = imagePoints.map(_.y).max
    val minX = imagePoints.map(_.x).min
    val maxX = imagePoints.map(_.x).max
    // lengths in each direction
    // minimal uncertainty in image is always half a pixel positioning error
    (math.max(0.5, maxX - minX), math.max(0.5, maxY - minY))
  }

  case class ImageFiducialLikelihood(measurementXSdev: Double,
                                     measurementYSdev: Double,
                                     measurementZSdev: Double,
                                     detectedFiducials: IndexedSeq[Fiducial],
                                     cameraModel: CameraPinholeBrown)
    extends DistributionEvaluator[PoseWithFocalLength] {

    override def logValue(sample: PoseWithFocalLength): Double = {
      // get the camera from the pose we introduced
      val camera = PoseWithFocalLengthToCamera(sample, cameraModel)

      // total transform from world onto image plane, including perspective camera transform
      val worldToImage: Point3D_F64 => Point2D_F64 = PerspectiveOps.createWorldToPixel(
        camera.cameraModel,
        camera.worldToCamera).transform

      val fidLikelihoods: IndexedSeq[Double] = detectedFiducials.map { fid: Fiducial =>
        // fiducial 2d image position of all 4 corners
        val controlsInImage = fid.controls2D

        // fiducial 3d world positions of all 4 corners
        val worldPosition = YuMiReferenceFiducials.referenceFiducialTransforms(fid.id)
        val refControls = fid.controlsInWorld(worldPosition)

        // evaluate distance
        val fiducialLH: Double = refControls.zip(controlsInImage).map {
          case (refPoint, observedPoint) =>
            // project reference into image
            val projectedRef: Point2D_F64 = worldToImage(refPoint)
            // pixel uncertainties of each corner in the image plane
            val (pixelXSdev, pixelYSdev) = projectWorldUncertaintyToImage(
              measurementXSdev,
              measurementYSdev,
              measurementZSdev,
              worldToImage,
              refPoint)
            // evaluate difference to observed point with Gaussian model
            val logLHX = GaussianEvaluator(projectedRef.x, pixelXSdev).logValue(observedPoint.x)
            val logLHY = GaussianEvaluator(projectedRef.y, pixelYSdev).logValue(observedPoint.y)
            logLHX + logLHY
        }.sum
        // total likelihood of this fiducial (all corner cond. independent)
        fiducialLH
      }
      // total likelihood of all fiducials (all fiducials cond. independent)
      fidLikelihoods.sum
    }
  }

  class ProposalLogger() extends AcceptRejectLogger[PoseWithFocalLength] {
    var accepted: Int = 0
    var rejected: Int = 0

    var acceptedMap: mutable.Map[String, Int] = mutable.Map.empty[String, Int].withDefaultValue(0)
    var rejectedMap: mutable.Map[String, Int] = mutable.Map.empty[String, Int].withDefaultValue(0)

    override def accept(current: PoseWithFocalLength,
                        sample: PoseWithFocalLength,
                        generator: ProposalGenerator[PoseWithFocalLength],
                        evaluator: DistributionEvaluator[PoseWithFocalLength]): Unit = {
      //println(f"Sample has been ACCEPTED: proposal=$sample (from current=$current)")
      acceptedMap(generator.toString) += 1
      accepted += 1
    }

    override def reject(current: PoseWithFocalLength,
                        sample: PoseWithFocalLength,
                        generator: ProposalGenerator[PoseWithFocalLength],
                        evaluator: DistributionEvaluator[PoseWithFocalLength]): Unit = {
      //println(f"Sample has been REJECTED: proposal=$sample (from current=$current)")
      rejectedMap(generator.toString) += 1
      rejected += 1
    }
  }
  
  def SphericalGeomStats(angles: Seq[Double]): (Double, Double) = {
    // transform each angle to x, y
    def toXY(angle: Double): (Double, Double) = (math.cos(angle), math.sin(angle))
    val XYAngles = angles.map(toXY)
    val xCoord = XYAngles.map(CartCoordinates => CartCoordinates._1)
    val yCoord = XYAngles.map(CartCoordinates => CartCoordinates._2)

    // average of [x, y]
    val xAnglesMean = xCoord.sum / xCoord.size
    val yAnglesMean = yCoord.sum / yCoord.size

    // standard deviation of [x, y]. Std Dev like sqrt(-2*ln(Module of mean of samples))
    val AnglesStDev = math.sqrt(-math.log(math.pow(xAnglesMean,2) + math.pow(yAnglesMean,2)))

    // transform [x, y] -> angle
    def toAngle(x: Double, y: Double): Double = {
      if (math.atan2(y, x) > 0) {
        math.atan2(y, x)
      }
      else {
        math.atan2(y, x) + 2 * math.Pi
      }
    }
    (toAngle(xAnglesMean, yAnglesMean), AnglesStDev)
  }

  def EuclideanGeomStats(values: Seq[Double]): (Double, Double) = {
    // average of [x, y]
    val Mean = values.sum / values.size

    // standard deviation of [x, y]. Variance like E[x^2]/N - M[x]^2
    val valuesSq = values.map(value => math.pow(value,2))
    val StDev = math.sqrt(valuesSq.sum / valuesSq.size - math.pow(Mean, 2))
    (Mean, StDev)
  }

  // get camera with the focal length parameters from the new proposal
  def PoseWithFocalLengthToCamera(PoseWithFocalLength: PoseWithFocalLength, cameraModel: CameraPinholeBrown): CameraInTheWorld = {
    val currentPose = Pose(PoseWithFocalLength.location, PoseWithFocalLength.orientation)
    val newCameraModel: CameraPinholeBrown = new CameraPinholeBrown(
      PoseWithFocalLength.focallengthX, PoseWithFocalLength.focallengthY,
      cameraModel.skew, cameraModel.cx, cameraModel.cy, cameraModel.width, cameraModel.height
    )
    newCameraModel.radial = cameraModel.radial.clone()
    newCameraModel.t1 = cameraModel.getT1
    newCameraModel.t2 = cameraModel.getT2

    CameraInTheWorld.fromCamera(currentPose.toSe3, newCameraModel)
  }

  def main(args: Array[String]): Unit = {
    val image = streamCamera.nextGrayImage

    val proposalBroadTSdev: Double = 10
    val proposalBroadRSdev: Double = 0.1
    val proposalNarrowTSdev: Double = 1
    val proposalNarrowRSdev: Double = 0.01
    val FocalLengthProposalSdev: Double = 15
    val measurementXSdev: Double = 20
    val measurementYSdev: Double = 20
    val measurementZSdev: Double = 20

    // getting the IDs of reference fiducials
    val refIds = YuMiReferenceFiducials.referenceIds
    // detecting all fiducials that match the large size. In our case reference + objects are both large
    val fidSqBinaryLarge = Fiducial.findAllFiducials(image, Fiducial.defaultSquareBinaryDetector(cameraModel, 47.0))
    // from all those fiducials, take only the ones that match the reference IDs (to find the camera)
    val refFids = fidSqBinaryLarge.filter { fid => refIds.contains(fid.id) }
    // object fiducials set (no use for now)
    val objFids = fidSqBinaryLarge.filter { fid => !refIds.contains(fid.id) && fid.id != MagicWand.wandId }

    val likelihoodEvaluator: DistributionEvaluator[PoseWithFocalLength] = ImageFiducialLikelihood(
      measurementXSdev, measurementYSdev, measurementZSdev,
      refFids,
      cameraModel
    )

    // 4 proposals. A broad (bigger jumps) and a narrow (local exporation) for T and R
    val proposalBroadT: ProposalGenerator[PoseWithFocalLength]
      with SymmetricTransitionRatio[PoseWithFocalLength] = CameraTProposal(proposalBroadTSdev)
    val proposalBroadR: ProposalGenerator[PoseWithFocalLength]
      with SymmetricTransitionRatio[PoseWithFocalLength] = CameraRProposal(proposalBroadRSdev)
    val proposalNarrowT: ProposalGenerator[PoseWithFocalLength]
      with SymmetricTransitionRatio[PoseWithFocalLength] = CameraTProposal(proposalNarrowTSdev)
    val proposalNarrowR: ProposalGenerator[PoseWithFocalLength]
      with SymmetricTransitionRatio[PoseWithFocalLength] = CameraRProposal(proposalNarrowRSdev)
    val proposalFocalLength: ProposalGenerator[PoseWithFocalLength]
      with SymmetricTransitionRatio[PoseWithFocalLength] = IntrinsicProposal(FocalLengthProposalSdev)

    val mixedProposal: ProposalGenerator[PoseWithFocalLength]
      with SymmetricTransitionRatio[PoseWithFocalLength] = MixtureProposal(
        0.25 *: proposalBroadT + 0.25 *: proposalBroadR + 0.25 *: proposalNarrowT + 0.25 *: proposalNarrowR + 0.0 *: proposalFocalLength
    )

    val mh = Metropolis(mixedProposal, likelihoodEvaluator)

    // finding camera translation vector and rotation matrix
    val camera = CameraInTheWorld.findCameraPositionUsingManyFiducials(
      refFids,
      YuMiReferenceFiducials.referenceFiducialTransforms,
      cameraModel,
      //MostStableEstimation,
      AverageEstimation,
      //RobustEstimation(maxIteration = 100, inlierThreshold = 4.0)
      )

    // obtain the CamParameters we need from the camera
    camera match {
      case Some(camera: CameraInTheWorld) =>
        // translation
        val x = camera.centerTranslation.getX
        val y = camera.centerTranslation.getY
        val z = camera.centerTranslation.getZ

        // rotation
        // transform of the matrix to a square matrix
        val OrientationMatrix = GeoUtils.boofToScalismo3D(camera.orientation)
        // determination of Euler angles (phi, theta, psi) from the retrieved rotation matrix
        val Angles_3D = LandmarkRegistration.rotMatrixToEulerAngles(OrientationMatrix.toBreezeMatrix)

        val xAngle = Angles_3D(2)
        val yAngle = Angles_3D(1)
        val zAngle = Angles_3D(0)

        val focallengthX = cameraModel.fx
        val focallengthY = cameraModel.fy

        val initialCamEstimate = PoseWithFocalLength(x, y, z, xAngle, yAngle, zAngle, focallengthX, focallengthY)
        val logger = new ProposalLogger()

        //(500, 40, 1200, 2.8, 0, 0), (400, 50, 1000, 3.1, 6, 0.1), (x, y, z, xAngle, yAngle, zAngle)
        val samples = mh.iterator(initialCamEstimate, logger).take(10000).toIndexedSeq

        val posteriorSamples: IndexedSeq[PoseWithFocalLength] = samples.drop(5000).take(5000)

        // calculate the mean and standard deviation of the samples
        // translation coordinates.
        val xValues = posteriorSamples.map{case PoseWithFocalLength(Location(x, _, _), Orientation(_, _, _),_,_) => x}
        val yValues = posteriorSamples.map{case PoseWithFocalLength(Location(_, y, _), Orientation(_, _, _),_,_) => y}
        val zValues = posteriorSamples.map{case PoseWithFocalLength(Location(_, _, z), Orientation(_, _, _),_,_) => z}
        val (xMean, xStDev) = EuclideanGeomStats(xValues)
        val (yMean, yStDev) = EuclideanGeomStats(yValues)
        val (zMean, zStDev) = EuclideanGeomStats(zValues)

        // rotation coordinates.
        val xAngles = posteriorSamples.map(pose => pose.orientation.xAngle)
        val yAngles = posteriorSamples.map(pose => pose.orientation.yAngle)
        val zAngles = posteriorSamples.map(pose => pose.orientation.zAngle)
        val (xAnglesMean, xAnglesStDev) = SphericalGeomStats(xAngles)
        val (yAnglesMean, yAnglesStDev) = SphericalGeomStats(yAngles)
        val (zAnglesMean, zAnglesStDev) = SphericalGeomStats(zAngles)

        // focal length
        val fx = posteriorSamples.map{case PoseWithFocalLength(Location(_, _, _), Orientation(_, _, _),fx,_) => fx}
        val fy = posteriorSamples.map{case PoseWithFocalLength(Location(_, _, _), Orientation(_, _, _),_,fy) => fy}
        val (fxMean, fxStDev) = EuclideanGeomStats(fx)
        val (fyMean, fyStDev) = EuclideanGeomStats(fy)


        println(f"accepted = ${logger.accepted} and rejected = ${logger.rejected}")
        println(f"acceptedMap = ${logger.acceptedMap}")
        println(f"rejectedMap = ${logger.rejectedMap}")

        println(initialCamEstimate)
        /*
        println("Location x: " + samples.map(pose => pose.location.x))
        println("Location y: " + samples.map(pose => pose.location.y))
        println("Location z: " + samples.map(pose => pose.location.z))
        println("Orientation xA: " + samples.map(pose => pose.orientation.xAngle))
        println("Orientation yA: " + samples.map(pose => pose.orientation.yAngle))
        println("Orientation zA: " + samples.map(pose => pose.orientation.zAngle))
         */

        println(f"Initial focal length: ${focallengthX}, ${focallengthY}")
        println(f"All the results are shown in the order: x, y, z, xAngle, yAngle, zAngle")
        println(f"The Camera is located in:")
        println(f"Means: ${xMean}, ${yMean}, ${zMean}, " +
          f"${xAnglesMean}, ${yAnglesMean}, ${zAnglesMean}")
        println(f"StdDev: ${xStDev}, ${yStDev}, ${zStDev}, " +
          f"${xAnglesStDev}, ${yAnglesStDev}, ${zAnglesStDev}")
        println(f"The Camera has the following focal length:")
        println(f"Mean: ${fxMean}, ${fyMean}")
        println(f"StdDev: ${fxStDev}, ${fyStDev}")

        // propagation of camera uncertainty to object uncertainty
        // get all cameras from the location samples sequence obtained
        val cameras: IndexedSeq[CameraInTheWorld] = samples.map(pose => PoseWithFocalLengthToCamera(pose, cameraModel))

        println("In process: estimation of the pose of the detected fiducials/objects")
        val objectsForCam: IndexedSeq[IndexedSeq[VisualObject]] =
          // case focal length is constant. Initial cameraModel maintained
          if (logger.acceptedMap(proposalFocalLength.toString) == 0) {
            // fiducials carry the camera model internally to provide an estimated pose, and there's no need to change it now
            val objectsForCam: IndexedSeq[IndexedSeq[VisualObject]] = cameras.map(
              (camera: CameraInTheWorld) => VisualObject.fromFiducials(objFids, camera)
            )
            objectsForCam
          }
          // case focal length independent to avoid calibration step
          else {
            // need to redetect and keep detector to reconfigure for each intrinsic setting
            val fiducialDetector = Fiducial.defaultSquareBinaryDetector(cameraModel, 47.0)
            fiducialDetector.detect(image)

            val objectsForCam: IndexedSeq[IndexedSeq[VisualObject]] = for (camera <- cameras) yield {
              // need to update the camera model used by this fiducial (internally)
              // reconfiguration of the detector with the new camera model
              val lensDistortion = new LensDistortionBrown(camera.cameraModel)
              fiducialDetector.setLensDistortion(lensDistortion, camera.cameraModel.width, camera.cameraModel.height)

              // re-estimation of fiducials pose
              val allFiducials = Fiducial.fromAllDetections(fiducialDetector)
              val objFids = allFiducials.filter { fid => !refIds.contains(fid.id) && fid.id != MagicWand.wandId }

              // save all the visual object pose of all detected objects, for this specific camera
              objFids.map(fid => VisualObject.fromFiducial(fid, camera))
            }
            objectsForCam
          }
        println(s"Calculation of object statistics")

        // 1: create long list of all objects (flatten: unpack inner lists to 1 large list)
        val allObjectSamples: IndexedSeq[VisualObject] = objectsForCam.flatten
        // 2: group by the object ids to get corresponding groups of samples of the same object
        val objSamples = allObjectSamples.groupBy(obj => obj.id)
        // 3: for each id / object run the statistics analysis
        for (objId <- objSamples.keys) {
          // all samples belonging to this object id
          val visObjSamples = objSamples(objId)
          // mapping the Visual Object to a Pose (with Euler angles (phi, theta, psi))
          val objectPoses = visObjSamples.map { case VisualObject(location, orientation, _) =>
            val orientationMatrix = GeoUtils.boofToScalismo3D(orientation)
            val Angles_3D = LandmarkRegistration.rotMatrixToEulerAngles(orientationMatrix.toBreezeMatrix)
            Pose(Location(location.x, location.y, location.z), Orientation(Angles_3D(2), Angles_3D(1), Angles_3D(0)))
          }

          val posteriorObjPoses: IndexedSeq[Pose] = objectPoses.drop(19).take(19)
          val fidXValues = posteriorObjPoses.map(pose => pose.location.x)
          val fidYValues = posteriorObjPoses.map(pose => pose.location.y)
          val fidZValues = posteriorObjPoses.map(pose => pose.location.z)
          val (fidXMean, fidXStDev) = EuclideanGeomStats(fidXValues)
          val (fidYMean, fidYStDev) = EuclideanGeomStats(fidYValues)
          val (fidZMean, fidZStDev) = EuclideanGeomStats(fidZValues)

          val fidXAngles = posteriorObjPoses.map(pose => pose.orientation.xAngle)
          val fidYAngles = posteriorObjPoses.map(pose => pose.orientation.yAngle)
          val fidZAngles = posteriorObjPoses.map(pose => pose.orientation.zAngle)
          val (fidXAnglesMean, fidXAnglesStDev) = SphericalGeomStats(fidXAngles)
          val (fidYAnglesMean, fidYAnglesStDev) = SphericalGeomStats(fidYAngles)
          val (fidZAnglesMean, fidZAnglesStDev) = SphericalGeomStats(fidZAngles)

          println(f"Object fiducial with id: ${objId}")
          println(f"has the location:")
          println(f"${fidXMean} ${fidXStDev}")
          println(f"${fidYMean}  ${fidYStDev}")
          println(f"${fidZMean} ${fidZStDev}")
          println(f"${fidXAnglesMean} ${fidXAnglesStDev}")
          println(f"${fidYAnglesMean} ${fidYAnglesStDev}")
          println(f"${fidZAnglesMean} ${fidZAnglesStDev}")

        }

      case None =>
        logger.debug("no origin found")
    }
  }
}
