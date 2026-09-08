import { BrowserRouter, Route, Routes } from "react-router-dom";
import { Dashboard } from "./pages/Dashboard";
import { InstrumentDetail } from "./pages/InstrumentDetail";

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/instrument/:ticker" element={<InstrumentDetail />} />
      </Routes>
    </BrowserRouter>
  );
}
