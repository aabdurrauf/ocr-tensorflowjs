let src = cv.imread('canvasInput');
let dst = cv.Mat.zeros(src.rows, src.cols, cv.CV_8UC3);
cv.cvtColor(src, src, cv.COLOR_RGBA2GRAY, 0);
cv.threshold(src, src, 100, 200, cv.THRESH_BINARY);
let contours = new cv.MatVector();
let hierarchy = new cv.Mat();
let poly = new cv.MatVector();
cv.findContours(src, contours, hierarchy, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE);
// approximates each contour to polygon
for (let i = 0; i < contours.size(); ++i) {
    let tmp = new cv.Mat();
    let cnt = contours.get(i);
    // You can try more different parameters
    cv.approxPolyDP(cnt, tmp, 3, true);
    poly.push_back(tmp);
    cnt.delete(); tmp.delete();
}
// draw contours with random Scalar
for (let i = 0; i < contours.size(); ++i) {
    let color = new cv.Scalar(Math.round(Math.random() * 255), Math.round(Math.random() * 255),
                              Math.round(Math.random() * 255));
    cv.drawContours(dst, poly, i, color, 1, 8, hierarchy, 0);
}
cv.imshow('canvasOutput', dst);
src.delete(); dst.delete(); hierarchy.delete(); contours.delete(); poly.delete();



------------------------------------------------


let src = cv.imread('canvasInput');
let dst = cv.Mat.zeros(src.rows, src.cols, cv.CV_8UC3);
cv.cvtColor(src, src, cv.COLOR_RGBA2GRAY, 0);
cv.threshold(src, src, 177, 200, cv.THRESH_BINARY);
let contours = new cv.MatVector();
let hierarchy = new cv.Mat();
cv.findContours(src, contours, hierarchy, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE);
let cnt = contours.get(0);
// You can try more different parameters
let rotatedRect = cv.minAreaRect(cnt);
let vertices = cv.RotatedRect.points(rotatedRect);
let contoursColor = new cv.Scalar(255, 255, 255);
let rectangleColor = new cv.Scalar(255, 0, 0);
cv.drawContours(dst, contours, 0, contoursColor, 1, 8, hierarchy, 100);
// draw rotatedRect
for (let i = 0; i < 4; i++) {
    cv.line(dst, vertices[i], vertices[(i + 1) % 4], rectangleColor, 2, cv.LINE_AA, 0);
}
cv.imshow('canvasOutput', dst);
src.delete(); dst.delete(); contours.delete(); hierarchy.delete(); cnt.delete();


    // for (let i = 0; i < contours.size(); i++) {
    //   const contour = contours.get(i);

    //   // Approximate the contour to a polygon
    //   const epsilon = 0.1 * cv.arcLength(contour, true); // Adjust sensitivity
    //   const approx = new cv.Mat();
    //   cv.approxPolyDP(contour, approx, epsilon, true);

    //   // Check if the approximated shape has 4 corners (potential rectangle)
    //   if (approx.rows === 4) {
    //     const points = [];

    //     for (let j = 0; j < 4; j++) {
    //       points.push(
    //         new cv.Point(approx.data32S[j * 2], approx.data32S[j * 2 + 1])
    //       );
    //     }

    //     // Convert points to a format suitable for OpenCV
    //     const rectContour = cv.matFromArray(4, 1, cv.CV_32SC2, [
    //       points[0].x,
    //       points[0].y,
    //       points[1].x,
    //       points[1].y,
    //       points[2].x,
    //       points[2].y,
    //       points[3].x,
    //       points[3].y,
    //     ]);

    //     const contoursVec = new cv.MatVector();
    //     contoursVec.push_back(rectContour);

    //     // Fill the rectangle
    //     const color = new cv.Scalar(255); // White color
    //     cv.fillPoly(output, contoursVec, color);

    //     // Clean up
    //     contoursVec.delete();
    //     rectContour.delete();
    //   }

    //   approx.delete();
    // }