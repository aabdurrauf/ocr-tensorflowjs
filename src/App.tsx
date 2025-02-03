import DetectBarCode from "./detectbarcode";
import DevOpenCV from "./devopencv";
import Ocr from "./Ocr";

function App() {
  return (
    <div className="container" style={{ height: "100vh", textAlign: "center" }}>
      {/* <Ocr /> */}
      {/* <DevOpenCV /> */}
      <DetectBarCode />
    </div>
  );
}

export default App;
