import {
  GraphModel,
  loadGraphModel,
  browser,
  scalar,
  squeeze,
  unstack,
  argMax,
  concat,
  softmax,
  Tensor,
  Tensor2D,
  Tensor3D,
  tidy,
} from "@tensorflow/tfjs";
import { MutableRefObject } from "react";
import { ModelConfig } from "../common/types";
import {
  DET_MEAN,
  DET_STD,
  MIN_RATIO,
  REC_MEAN,
  REC_STD,
  VOCAB,
} from "../common/constants";
import cv from "@techstark/opencv-js";
import { AnnotationShape, Stage } from "react-mindee-js";

import { Layer } from "konva/lib/Layer";
import { chunk } from "underscore";

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

/// ---------- relating bounding box extractions --------------

export const getDetectedBoundingBoxes = async ({
  heatmapContainer,
  detectionModel,
  imgRef,
  size,
}: {
  heatmapContainer: HTMLCanvasElement | null;
  detectionModel: GraphModel | null;
  imgRef: HTMLImageElement;
  size: [number, number];
}) =>
  new Promise<void>(async (resolve) => {
    if (!heatmapContainer || !detectionModel) {
      return;
    }
    heatmapContainer!.width = imgRef.width;
    heatmapContainer!.height = imgRef.height;

    let prediction: Tensor | null = null;

    try {
      const pred = tidy(() => {
        const tensor = getImageTensorForDetectionModel(imgRef, size);

        prediction = detectionModel.execute(tensor) as Tensor;
        const squeezedPrediction = squeeze(prediction, [0]);
        
        let finalPrediction: Tensor2D | Tensor3D;
        if (squeezedPrediction.shape.length === 2) {
          // Grayscale image: Add the 3rd dimension (for RGB channels)
          finalPrediction = squeezedPrediction
            .expandDims(-1)
            .tile([1, 1, 3]) as Tensor3D;
        } else {
          // Already a 3D tensor
          finalPrediction = squeezedPrediction as Tensor3D;
        }

        return finalPrediction;
      });

      // Convert to canvas
      await browser.toPixels(pred, heatmapContainer);
      pred.dispose();
    } catch (error) {
      console.error("Error during detection:", error);
    } finally {
      // Dispose of prediction if it exists
      if (prediction) {
        (prediction as Tensor).dispose();
      }
      resolve();
    }
  });

export const getImageTensorForDetectionModel = (
  imageObject: HTMLImageElement,
  size: [number, number]
) => {
  return tidy(() => {
    let tensor = browser
      .fromPixels(imageObject)
      .resizeNearestNeighbor(size)
      .toFloat();
    let mean = scalar(255 * DET_MEAN);
    let std = scalar(255 * DET_STD);
    return tensor.sub(mean).div(std).expandDims();
  });
};

export const extractBoundingBoxes = (size: [number, number]) => {
  let src = cv.imread("heatmap");
  cv.cvtColor(src, src, cv.COLOR_RGBA2GRAY, 0);
  cv.threshold(src, src, 77, 255, cv.THRESH_BINARY);
  cv.morphologyEx(src, src, cv.MORPH_OPEN, cv.Mat.ones(2, 2, cv.CV_8U));

  let contours = new cv.MatVector();
  let hierarchy = new cv.Mat();
  // You can try more different parameters
  cv.findContours(
    src,
    contours,
    hierarchy,
    cv.RETR_EXTERNAL,
    cv.CHAIN_APPROX_SIMPLE
  );
  // draw contours with random Scalar
  const boundingBoxes = [];
  // @ts-ignore
  for (let i = 0; i < contours.size(); ++i) {
    const contourBoundingBox = cv.boundingRect(contours.get(i));
    if (
      contourBoundingBox.width > 2 &&
      contourBoundingBox.height > 2 &&
      contourBoundingBox.height / contourBoundingBox.width < MIN_RATIO
    ) {
      const transformedBox = transformBoundingBox(contourBoundingBox, i, size);
      if (transformedBox !== null) {
        boundingBoxes.unshift(transformedBox);
      }
    }
  }

  src.delete();
  contours.delete();
  hierarchy.delete();
  // console.log("boundingBoxes: ", boundingBoxes);
  return boundingBoxes;
};

function clamp(number: number, size: number) {
  return Math.max(0, Math.min(number, size));
}

export const transformBoundingBox = (
  contour: any,
  id: number,
  size: [number, number]
): AnnotationShape => {
  let offset =
    (contour.width * contour.height * 1.8) /
    (2 * (contour.width + contour.height));
  const p1 = clamp(contour.x - offset, size[1]) - 1;
  const p2 = clamp(p1 + contour.width + 2 * offset, size[1]) - 1;
  const p3 = clamp(contour.y - offset, size[0]) - 1;
  const p4 = clamp(p3 + contour.height + 2 * offset, size[0]) - 1;
  return {
    id,
    config: {
      stroke: "#f54242",
    },
    coordinates: [
      [p1 / size[1], p3 / size[0]],
      [p2 / size[1], p3 / size[0]],
      [p2 / size[1], p4 / size[0]],
      [p1 / size[1], p4 / size[0]],
    ],
  };
};

/// ---------- relating words extractions --------------

export const extractWords = async ({
  recognitionModel,
  stage,
  size,
}: {
  recognitionModel: GraphModel | null;
  stage: Stage;
  size: [number, number];
}) => {
  const crops = (await getCrops({ stage })) as Array<{
    id: string;
    crop: HTMLImageElement;
    color: string;
  }>;
  const chunks = chunk(crops, 16);
  return Promise.all(
    chunks.map(
      (chunk) =>
        new Promise(async (resolve) => {
          const words = await extractWordsFromCrop({
            recognitionModel,
            crops: chunk.map((elem) => elem.crop),
            size,
          });
          const collection = words?.map((word: any, index: any) => ({
            ...chunk[index],
            words: word ? [word] : [],
          }));
          resolve(collection);
        })
    )
  );
};

export const getCrops = ({ stage }: { stage: Stage }) => {
  if (!stage) {
    throw new Error(
      "Stage is undefined or null. Ensure you pass a valid Stage object."
    );
  }

  const layer = stage.findOne<Layer>("#shapes-layer");

  if (!layer) {
    throw new Error("Layer with ID '#shapes-layer' not found in the stage.");
  }
  const polygons = layer.find(".shape");

  return Promise.all(
    polygons
      .map((polygon: any) => {
        const clientRect = polygon.getClientRect();

        const ratio = clientRect.height / clientRect.width;
        if (ratio > MIN_RATIO) {
          return null;
        }

        return new Promise((resolve) => {
          stage.toImage({
            ...clientRect,
            quality: 5,
            pixelRatio: 10,
            callback: (value: HTMLImageElement) => {
              resolve({
                id: polygon.id(),
                crop: value,
                color: polygon.getAttr("stroke"),
              });
            },
          });
        });
      })
      .filter(Boolean)
  );
};

export const extractWordsFromCrop = async ({
  recognitionModel,
  crops,
  size,
}: {
  recognitionModel: GraphModel | null;
  crops: any;
  size: [number, number];
}) => {
  if (!recognitionModel) {
    return;
  }

  const blank = 126;
  let tensor: Tensor | null = null;
  let predictions: Tensor | Tensor[] | null = null;

  try {
    tensor = tidy(() => {
      return getImageTensorForRecognitionModel(crops, size);
    });
    // console.log("tensor: ", tensor);
    const predictions = await recognitionModel.executeAsync(tensor);

    const words = tidy(() => {
      const wordsTemp: string[] = [];
      let predictionTensor: Tensor;
      if (Array.isArray(predictions)) {
        predictionTensor = predictions[0];
      } else {
        predictionTensor = predictions!;
      }

      const probabilities = softmax(predictionTensor, -1);
      const bestPath = unstack(argMax(probabilities, -1), 0);

      for (const sequence of bestPath) {
        let collapsed = "";
        let added = false;
        const values = sequence.dataSync();
        const arr = Array.from(values);

        for (const k of arr as number[]) {
          if (k === blank) {
            added = false;
          } else if (k !== blank && !added) {
            collapsed += VOCAB[k];
            added = true;
          }
        }

        wordsTemp.push(collapsed);
        return wordsTemp;
      }
    });
    return words;
  } catch (error) {
    console.error("Error during word extraction:", error);
    return [];
  } finally {
    if (tensor) tensor.dispose();
    if (predictions) {
      if (Array.isArray(predictions)) {
        (predictions as Tensor[]).forEach((pred) => pred.dispose());
      } else {
        (predictions as Tensor).dispose();
      }
    }
  }
};

export const getImageTensorForRecognitionModel = (
  crops: HTMLImageElement[],
  size: [number, number]
) => {
  const list = crops.map((imageObject) => {
    let h = imageObject.height;
    let w = imageObject.width;
    let resize_target: any;
    let padding_target: any;
    let aspect_ratio = size[1] / size[0];
    if (aspect_ratio * h > w) {
      resize_target = [size[0], Math.round((size[0] * w) / h)];
      padding_target = [
        [0, 0],
        [0, size[1] - Math.round((size[0] * w) / h)],
        [0, 0],
      ];
    } else {
      resize_target = [Math.round((size[1] * h) / w), size[1]];
      padding_target = [
        [0, size[0] - Math.round((size[1] * h) / w)],
        [0, 0],
        [0, 0],
      ];
    }
    return browser
      .fromPixels(imageObject)
      .resizeNearestNeighbor(resize_target)
      .pad(padding_target, 0)
      .toFloat()
      .expandDims();
  });
  const tensor = concat(list);
  let mean = scalar(255 * REC_MEAN);
  let std = scalar(255 * REC_STD);
  return tensor.sub(mean).div(std);
};
