import React, { useEffect, useRef, useState } from "react";
import Webcam from "react-webcam";
import { DET_CONFIG, RECO_CONFIG, CODE_LEN, INTERVAL } from "./common/constants";
import { GraphModel, backend, disposeVariables } from "@tensorflow/tfjs";
import {
  loadDetectionModel,
  loadRecognitionModel,
  getDetectedBoundingBoxes,
  extractBoundingBoxes,
  extractWords,
} from "./utils/helper";
import { AnnotationData, AnnotationViewer, Stage } from "react-mindee-js";
import HeatMap from "./components/HeatMap";
import { Word } from "./common/types";
import { flatten } from "underscore";

export default function Ocr() {
  const [modelLoaded, setModelLoaded] = useState(false);
  const recoConfig = RECO_CONFIG.crnn_vgg16_bn;
  const detConfig = DET_CONFIG.db_mobilenet_v2;

  const [annotationData, setAnnotationData] = useState<AnnotationData>({
    image: null,
  });
  const [predictedWords, setPredictedWords] = useState<string>(
    "Bir yazı gösterin..."
  );
  const [words, setWords] = useState<Word[]>([]);

  useEffect(() => {
    if (words && words.length > 0) {
      const selectedWord = words.reduce((prev, current) => {
        const currentWord = current.words[0] || "";
        return currentWord.length === CODE_LEN ? currentWord : prev;
      }, "");
      console.log("Longest word detected:", selectedWord);

      setPredictedWords(selectedWord);
      // if (predictedWords.length < longestWord.length) {
      //   setPredictedWords(longestWord);
      // }
    }
  }, [words]);

  const recognitionModel = useRef<GraphModel | null>(null);
  const detectionModel = useRef<GraphModel | null>(null);
  const webcamRef = useRef<any | null>(null);
  const imageObject = useRef<HTMLImageElement>(new Image());
  const heatmapContainer = useRef<HTMLCanvasElement | null>(null);
  const annotationStage = useRef<Stage | null>();

  useEffect(() => {
    setAnnotationData({ image: null });
    loadDetectionModel({ detectionModel, detConfig });
    loadRecognitionModel({ recognitionModel, recoConfig });
    setModelLoaded(true);
  }, [detConfig, recoConfig]);

  useEffect(() => {
    if (heatmapContainer.current) {
      const context = heatmapContainer.current.getContext("2d", {
        willReadFrequently: true,
      });
      context?.clearRect(
        0,
        0,
        heatmapContainer.current.width,
        heatmapContainer.current.height
      );
    }
  }, []);

  useEffect(() => {
    if (!modelLoaded) return;

    const detectTexts = async (): Promise<void> => {
      const imageSrc = webcamRef.current.getScreenshot();
      if (!imageSrc) return;

      return new Promise<void>((resolve) => {
        imageObject.current.onload = async () => {
          try {
            await getDetectedBoundingBoxes({
              heatmapContainer: heatmapContainer.current,
              detectionModel: detectionModel.current,
              imgRef: imageObject.current,
              size: [detConfig.height, detConfig.width],
            });
            getBoundingBoxes();
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
    let lastTime = 0;
    const interval = INTERVAL;

    const nextTick = (timestamp: number) => {
      if (timestamp - lastTime >= interval) {
        lastTime = timestamp;
        detectTexts();
        disposeVariables(); // Dispose of any TensorFlow.js variables in memory
        backend().dispose(); // Dispose of the current backend, which clears WebGL resources
      }
      handle = requestAnimationFrame(nextTick);
    };

    handle = requestAnimationFrame(nextTick);
    return () => {
      cancelAnimationFrame(handle);
    };
  }, [modelLoaded]);

  const setAnnotationStage = (stage: Stage) => {
    annotationStage.current = stage;
  };

  const getBoundingBoxes = () => {
    const boundingBoxes = extractBoundingBoxes([
      detConfig.height,
      detConfig.width,
    ]);
    setAnnotationData({
      image: imageObject.current.src,
      shapes: boundingBoxes,
    });
    setTimeout(getWords, 1000);
  };

  const getWords = async () => {
    const words = (await extractWords({
      recognitionModel: recognitionModel.current,
      stage: annotationStage.current!,
      size: [recoConfig.height, recoConfig.width],
    })) as Word[];
    setWords(flatten(words));
  };

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        padding: "10px",
        fontFamily: "Arial, sans-serif",
      }}
    >
      <header style={{ marginBottom: "20px" }}>
        <h2>Real-Time OCR</h2>
      </header>
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          gap: "25px",
        }}
      >
        {/* Webcam Section */}
        <div
          style={{
            flex: 1,
            display: "flex",
            justifyContent: "center",
            alignItems: "center",
          }}
        >
          <Webcam
            ref={webcamRef}
            // mirrored
            screenshotFormat="image/jpeg"
            style={{
              width: "400px", // Set the desired width
              height: "300px", // Set the desired height
            }}
          />
        </div>

        <HeatMap heatMapContainerRef={heatmapContainer} />
        {annotationData && (
          <div
            style={{
              width: "400px",
              height: "300px",
              display: "flex",
              justifyContent: "center",
            }}
          >
            <AnnotationViewer
              data={annotationData}
              getStage={setAnnotationStage}
              style={{
                width: "100%",
                height: "100%",
                objectFit: "cover",
              }}
            />
          </div>
        )}
      </div>

      {/* Loading Message */}
      {!modelLoaded && (
        <div
          style={{
            textAlign: "center",
            fontSize: "16px",
          }}
        >
          Loading OCR model...
        </div>
      )}
      <p>{predictedWords}</p>
    </div>
  );
}
