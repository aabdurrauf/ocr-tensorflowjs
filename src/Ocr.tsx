import React, { useEffect, useRef, useState } from "react";
import Webcam from "react-webcam";
import { DET_CONFIG, RECO_CONFIG } from "./common/constants";
import { GraphModel } from "@tensorflow/tfjs";
import {
  loadDetectionModel,
  loadRecognitionModel,
  getDetectedText,
} from "./utils/helper";
import { AnnotationData, AnnotationViewer } from "react-mindee-js";

export default function Ocr() {
  const [modelLoaded, setModelLoaded] = useState(false);
  const recoConfig = RECO_CONFIG.crnn_vgg16_bn;
  const detConfig = DET_CONFIG.db_mobilenet_v2;

  const recognitionModel = useRef<GraphModel | null>(null);
  const detectionModel = useRef<GraphModel | null>(null);
  const webcamRef = useRef<any | null>(null);
  const imageObject = useRef<HTMLImageElement>(new Image());
  const textImgRef = useRef<any | null>(null);

  useEffect(() => {
    loadDetectionModel({ detectionModel, detConfig });
    loadRecognitionModel({ recognitionModel, recoConfig });
    setModelLoaded(true);
  }, [detConfig]);

  useEffect(() => {
    if (!modelLoaded) return;

    const detectTexts = async (): Promise<void> => {
      const imageSrc = webcamRef.current.getScreenshot();
      if (!imageSrc) return;

      return new Promise<void>((resolve) => {
        imageObject.current.onload = async () => {
          try {
            // const img = cv.imread(imageObject.current);
            // console.log("img: ", img);

            // cv.imshow(textImgRef.current, img);

            await getDetectedText({
              detectionModel: detectionModel.current,
              imgRef: imageObject.current,
              size: [detConfig.height, detConfig.width],
            });

            // img.delete();
            resolve();
          } catch (error) {
            console.log(error);
            resolve();
          }
        };
        imageObject.current.src = imageSrc;
      });
    };

    let handle: any;
    const nextTick = () => {
      handle = requestAnimationFrame(async () => {
        await detectTexts();
        nextTick();
      });
    };
    nextTick();
    return () => {
      cancelAnimationFrame(handle);
    };
  }, [modelLoaded]);

  return (
    <div className="Ocr">
      <h2>Real-Time OCR</h2>
      <Webcam
        ref={webcamRef}
        className="webcam"
        mirrored
        screenshotFormat="image/jpeg"
      />
      {/* <img className="inputImage" alt="input" ref={imageObject} /> */}
      <AnnotationViewer data={[] as AnnotationData} />
      {!modelLoaded && <div>Loading OCR model...</div>}
    </div>
  );
}
