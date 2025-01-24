import { MutableRefObject } from "react";

interface Props {
  heatMapContainerRef: MutableRefObject<HTMLCanvasElement | null>;
}

export default function HeatMap({ heatMapContainerRef }: Props): JSX.Element {
  return (
    <canvas
      id="heatmap"
      ref={heatMapContainerRef}
      style={{ height: "35vh", margin: "auto", display: "none" }}
    />
  );
}
