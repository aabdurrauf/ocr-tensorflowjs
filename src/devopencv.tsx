import React, { useEffect, useState } from "react";
import cv from "@techstark/opencv-js"; // Import OpenCV.js

const DevOpenCV: React.FC = () => {
  const [cvReady, setCvReady] = useState(false);
  const imageSrcs = [
    `${process.env.PUBLIC_URL}/package_images/Aras_2.jpeg`,
    `${process.env.PUBLIC_URL}/package_images/Aras_3.jpeg`,
    `${process.env.PUBLIC_URL}/package_images/Aras.jpeg`,
    `${process.env.PUBLIC_URL}/package_images/MNG.jpeg`,
    `${process.env.PUBLIC_URL}/package_images/MNG_2.jpeg`,
    `${process.env.PUBLIC_URL}/package_images/MNG_3.jpeg`,
    `${process.env.PUBLIC_URL}/package_images/PTT.jpeg`,
    `${process.env.PUBLIC_URL}/package_images/SENDEO.jpeg`,
    `${process.env.PUBLIC_URL}/package_images/Trendyol Express.jpeg`,
    `${process.env.PUBLIC_URL}/package_images/Trendyol Express_3.jpeg`,
    `${process.env.PUBLIC_URL}/package_images/Yurtiçi_2.jpeg`,
    `${process.env.PUBLIC_URL}/package_images/Yurtiçi_3.jpeg`,
  ];

  useEffect(() => {
    if (cv) {
      console.log("OpenCV.js is ready!");
      setCvReady(true);
    } else {
      console.error("Failed to load OpenCV.js");
    }
  }, []);

  const processImage = () => {
    if (!cvReady) {
      console.error("OpenCV.js is not ready yet!");
      return;
    }

    imageSrcs.forEach((imageSrc, i) => {
      const imgElement = new Image();
      imgElement.src = imageSrc;
      imgElement.onload = () => processSingleImage(imgElement, i);
    });
  };

  const processSingleImage = (imgElement: HTMLImageElement, index: number) => {
    const src = cv.imread(imgElement);
    const gray = new cv.Mat();
    cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);

    // **Histogram Calculation**
    let hist = new cv.Mat();
    let mask = new cv.Mat();
    const channels = [0]; // Single channel (gray)
    const histSize = [256]; // Number of bins (256 for grayscale)
    const ranges = [0, 255]; // Pixel intensity range (0 to 255)

    // Convert the regular array into a MatVector
    const matVector = new cv.MatVector();
    matVector.push_back(gray); // Push the binary image into the MatVector

    // Note: No mask here (pass cv.Mat() if you don't need it)
    cv.calcHist(matVector, channels, mask, hist, histSize, ranges, false);

    // Plot histogram
    plotHistogram(src, hist, mask, histSize, `canvasHist${index}`);

    const histData = Array.from(hist.data32F);
    const sumData = histData.reduce((acc, val, index) => acc + val * index, 0);
    const sumPixel = src.rows * src.cols;
    const binThr = sumData / sumPixel;

    // **Binarization (Thresholding)**
    const binary_ori = new cv.Mat();
    cv.threshold(
      gray,
      binary_ori,
      binThr + 20,
      255,
      cv.THRESH_BINARY_INV + cv.THRESH_OTSU
    );

    const binary = new cv.Mat();
    // Perform morph operatiosn
    // ERODE
    let M = cv.Mat.ones(15, 1, cv.CV_8U);
    let anchor = new cv.Point(-1, -1);
    cv.erode(
      binary_ori,
      binary,
      M,
      anchor,
      1,
      cv.BORDER_CONSTANT,
      cv.morphologyDefaultBorderValue()
    );
    // OPEN
    const subBinary = new cv.Mat();
    M = cv.Mat.ones(1, 20, cv.CV_8U);
    cv.morphologyEx(
      binary,
      subBinary,
      cv.MORPH_OPEN,
      M,
      anchor,
      1,
      cv.BORDER_CONSTANT,
      cv.morphologyDefaultBorderValue()
    );
    // DILATE
    M = cv.Mat.ones(20, 20, cv.CV_8U);
    cv.dilate(
      subBinary,
      subBinary,
      M,
      anchor,
      1,
      cv.BORDER_CONSTANT,
      cv.morphologyDefaultBorderValue()
    );

    const subtracted = new cv.Mat();
    cv.subtract(binary, subBinary, subtracted, new cv.Mat(), -1);

    cv.resize(binary, binary, new cv.Size(300, 400), 0, 0, cv.INTER_AREA);
    cv.imshow(`canvasOutput${index}`, binary);

    const morp2 = new cv.Mat();
    cv.bitwise_not(subtracted, morp2);
    // OPEN
    M = cv.Mat.ones(4, 20, cv.CV_8U);
    cv.morphologyEx(
      morp2,
      morp2,
      cv.MORPH_OPEN,
      M,
      anchor,
      1,
      cv.BORDER_CONSTANT,
      cv.morphologyDefaultBorderValue()
    );
    // OPEN
    M = cv.Mat.ones(20, 20, cv.CV_8U);
    cv.morphologyEx(
      morp2,
      morp2,
      cv.MORPH_OPEN,
      M,
      anchor,
      1,
      cv.BORDER_CONSTANT,
      cv.morphologyDefaultBorderValue()
    );
    cv.bitwise_not(morp2, morp2);
    // OPEN
    M = cv.Mat.ones(10, 40, cv.CV_8U);
    cv.morphologyEx(
      morp2,
      morp2,
      cv.MORPH_OPEN,
      M,
      anchor,
      1,
      cv.BORDER_CONSTANT,
      cv.morphologyDefaultBorderValue()
    );
    // OPEN
    M = cv.Mat.ones(20, 20, cv.CV_8U);
    cv.morphologyEx(
      morp2,
      morp2,
      cv.MORPH_OPEN,
      M,
      anchor,
      1,
      cv.BORDER_CONSTANT,
      cv.morphologyDefaultBorderValue()
    );

    cv.resize(morp2, morp2, new cv.Size(300, 400), 0, 0, cv.INTER_AREA);
    cv.imshow(`canvasOther${index}`, morp2);

    const contours = new cv.MatVector();
    const hierarchy = new cv.Mat();
    // Find contours in the binary image
    cv.findContours(
      morp2,
      contours,
      hierarchy,
      cv.RETR_EXTERNAL, // Detect only external contours
      cv.CHAIN_APPROX_SIMPLE
    );

    // Create an empty image to hold the result
    const output = cv.Mat.zeros(binary.rows, binary.cols, cv.CV_8UC1);

    // Iterate through each contour
    for (let i = 0; i < contours.size(); i++) {
      const contour = contours.get(i);

      // Get the minimum area rectangle for the contour
      const rotatedRect = cv.minAreaRect(contour);
      const { center, size, angle } = rotatedRect;

      let takeRect = false;
      if (size.height > size.width) {
        if (size.height > 50) {
          takeRect = true;
        }
      } else {
        if (size.width > 50) {
          takeRect = true;
        }
      }
      if (takeRect) {
        // Compute the 4 corner points manually
        const rectPoints = [];

        const cosA = Math.cos((angle * Math.PI) / 180.0);
        const sinA = Math.sin((angle * Math.PI) / 180.0);

        const w = size.width / 2;
        const h = size.height / 2;

        rectPoints.push(
          new cv.Point(
            center.x - w * cosA - h * sinA,
            center.y - w * sinA + h * cosA
          )
        );
        rectPoints.push(
          new cv.Point(
            center.x + w * cosA - h * sinA,
            center.y + w * sinA + h * cosA
          )
        );
        rectPoints.push(
          new cv.Point(
            center.x + w * cosA + h * sinA,
            center.y + w * sinA - h * cosA
          )
        );
        rectPoints.push(
          new cv.Point(
            center.x - w * cosA + h * sinA,
            center.y - w * sinA - h * cosA
          )
        );

        // Convert rectPoints to cv.Mat
        const rectContour = cv.matFromArray(4, 1, cv.CV_32SC2, [
          rectPoints[0].x,
          rectPoints[0].y,
          rectPoints[1].x,
          rectPoints[1].y,
          rectPoints[2].x,
          rectPoints[2].y,
          rectPoints[3].x,
          rectPoints[3].y,
        ]);

        // Wrap rectContour in a MatVector
        const contoursVec = new cv.MatVector();
        contoursVec.push_back(rectContour);

        // Draw the rectangle
        const color = new cv.Scalar(255);
        cv.polylines(output, contoursVec, true, color, 2);

        // Clean up memory
        rectContour.delete();
        contoursVec.delete();
      }
    }

    cv.resize(output, output, new cv.Size(300, 400), 0, 0, cv.INTER_AREA);
    cv.imshow(`canvasFinal${index}`, output);

    // Clean up
    src.delete();
    gray.delete();
    binary.delete();
    hist.delete();
  };

  // **Histogram Plotting Function**
  const plotHistogram = (
    src: cv.Mat,
    hist: cv.Mat,
    mask: cv.Mat,
    histSize: number[],
    canvasId: string
  ) => {
    const scale = 2;
    let result = cv.minMaxLoc(hist, mask);
    let max = result.maxVal;
    let dst = cv.Mat.zeros(src.rows, histSize[0] * scale, cv.CV_8UC3);

    for (let i = 0; i < histSize[0]; i++) {
      let binVal = (hist.data32F[i] * src.rows) / max;
      let point1 = new cv.Point(i * scale, src.rows - 1);
      let point2 = new cv.Point((i + 1) * scale - 1, src.rows - binVal);
      cv.rectangle(
        dst,
        point1,
        point2,
        new cv.Scalar(255, 255, 255),
        cv.FILLED
      );
    }
    cv.resize(dst, dst, new cv.Size(300, 200), 0, 0, cv.INTER_AREA);
    cv.bitwise_not(dst, dst);
    cv.imshow(canvasId, dst);
  };

  return (
    <div>
      <div>
        <button onClick={processImage} disabled={!cvReady}>
          {cvReady ? "Detect Barcode" : "Loading OpenCV..."}
        </button>
      </div>
      <div>
        <h3>Original Images, Binarized Outputs & Histograms</h3>
        {imageSrcs.map((src, index) => (
          <div
            key={index}
            style={{
              display: "flex",
              alignItems: "center",
              gap: "20px",
              marginBottom: "20px",
            }}
          >
            {/* Left: Original Image */}
            <img src={src} alt={`Package ${index + 1}`} width="300px" />

            {/* Center: Processed Binary Image */}
            <canvas
              id={`canvasOutput${index}`}
              width="300"
              height="400"
              style={{ border: "1px solid black" }}
            ></canvas>

            {/* Right: Histogram Canvas */}
            <canvas
              id={`canvasOther${index}`}
              width="300"
              height="400"
              style={{ border: "1px solid black" }}
            ></canvas>

            {/* Right: Histogram Canvas */}
            <canvas
              id={`canvasFinal${index}`}
              width="300"
              height="400"
              style={{ border: "1px solid black" }}
            ></canvas>

            {/* Most Right: Histogram Canvas */}
            <canvas
              id={`canvasHist${index}`}
              width="300"
              height="200"
              style={{ border: "1px solid black" }}
            ></canvas>
          </div>
        ))}
      </div>
    </div>
  );
};

export default DevOpenCV;
