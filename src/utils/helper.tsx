import {
  GraphModel,
  loadGraphModel,
  browser,
  scalar,
  squeeze,
} from "@tensorflow/tfjs";
import { MutableRefObject } from "react";
import { ModelConfig } from "../common/types";
import { DET_MEAN, DET_STD } from "../common/constants";

export const loadDetectionModel = async ({
  detectionModel,
  detConfig,
}: {
  detectionModel: MutableRefObject<GraphModel | null>;
  detConfig: ModelConfig;
}) => {
  try {
    detectionModel.current = await loadGraphModel(detConfig.path);
  } catch (error) {
    console.log(error);
  }
};

export const loadRecognitionModel = async ({
  recognitionModel,
  recoConfig,
}: {
  recognitionModel: MutableRefObject<GraphModel | null>;
  recoConfig: ModelConfig;
}) => {
  try {
    recognitionModel.current = await loadGraphModel(recoConfig.path);
  } catch (error) {
    console.log(error);
  }
};

export const getDetectedText = async ({
  detectionModel,
  imgRef,
  size,
}: {
  detectionModel: GraphModel | null;
  imgRef: HTMLImageElement;
  size: [number, number];
}) =>
  new Promise(async (resolve) => {
    if (!detectionModel) {
      return;
    }
    let tensor = getImageTensorForDetectionModel(imgRef, size);
    let prediction: any = await detectionModel?.execute(tensor);
    // // @ts-ignore
    prediction = squeeze(prediction, [0]);
    if (Array.isArray(prediction)) {
      prediction = prediction[0];
    }
    // // @ts-ignore
    // await browser.toPixels(prediction, heatmapContainer);

    console.log("prediction: ", prediction);
    resolve("test");
  });

export const getImageTensorForDetectionModel = (
  imageObject: HTMLImageElement,
  size: [number, number]
) => {
  let tensor = browser
    .fromPixels(imageObject)
    .resizeNearestNeighbor(size)
    .toFloat();
  let mean = scalar(255 * DET_MEAN);
  let std = scalar(255 * DET_STD);
  return tensor.sub(mean).div(std).expandDims();
};
