import { StrictMode, useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter, Link, NavLink, Route, Routes } from "react-router-dom";
import Home from "./pages/Home";
import Replay from "./pages/Replay";
import Scorecard from "./pages/Scorecard";
import Learning from "./pages/Learning";
import Lab from "./pages/Lab";
import "./styles.css";

type Theme = "light" | "dark" | null;
const readTheme = (): Theme => { try { return localStorage.getItem("theme") as Theme; } catch { return null; } };

function ThemeToggle() {
  const [theme, setTheme] = useState<Theme>(readTheme);
  useEffect(() => {
    if (theme) document.documentElement.dataset.theme = theme; else delete document.documentElement.dataset.theme;
    try { if (theme) localStorage.setItem("theme", theme); else localStorage.removeItem("theme"); } catch { /* storage unavailable */ }
  }, [theme]);
  const isDark = theme ? theme === "dark" : !window.matchMedia("(prefers-color-scheme: light)").matches;
  return (
    <button className="icon-btn" aria-label={`Switch to ${isDark ? "light" : "dark"} theme`} onClick={() => setTheme(isDark ? "light" : "dark")}>
      {isDark ? "☀" : "☾"}
    </button>
  );
}

function App() {
  return (
    <BrowserRouter>
      <nav className="nav">
        <div className="wrap">
          <Link to="/" className="brand"><span className="brand-mark"><i /><i /><i /><i /></span><span className="brand-name">RL Trader</span></Link>
          <div className="nav-links">
            <NavLink to="/replay/SPY">Replay</NavLink>
            <NavLink to="/scorecard">Scorecard</NavLink>
            <NavLink to="/learning">Learning</NavLink>
            <NavLink to="/lab">Lab</NavLink>
          </div>
          <ThemeToggle />
        </div>
      </nav>
      <main>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/replay/:ticker?" element={<Replay />} />
          <Route path="/scorecard" element={<Scorecard />} />
          <Route path="/learning" element={<Learning />} />
          <Route path="/lab" element={<Lab />} />
          <Route path="*" element={<div className="wrap"><h1>Not found</h1></div>} />
        </Routes>
      </main>
      <footer className="footer"><div className="wrap">Research project, not investment advice. Results are out-of-sample backtests with transaction costs.</div></footer>
    </BrowserRouter>
  );
}

createRoot(document.getElementById("root")!).render(<StrictMode><App /></StrictMode>);
