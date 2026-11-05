
import {
  BrowserRouter,
  Routes,
  Route,
  Navigate,
  Outlet,
} from "react-router-dom";

import Sidebar from "./components/Sidebar";

// Global App Context
import { AppProvider } from "./context/AppContext";

/* ================= PUBLIC PAGES ================= */

import Landing from "./pages/Landing";
import Login from "./pages/Login";
import Signup from "./pages/Signup";
import ForgotPassword from "./pages/ForgotPassword";

/* ================= APP PAGES ================= */

import Dashboard from "./pages/Dashboard";
import Applications from "./pages/Applications";
import ApplicationDetails from "./pages/ApplicationDetails";
import AIAnalysis from "./pages/AIAnalysis";
import Infrastructure from "./pages/Infrastructure";
import Analytics from "./pages/Analytics";
import Forecasting from "./pages/Forecasting";
import Notifications from "./pages/Notifications";
import Profile from "./pages/Profile";
import Settings from "./pages/Settings";

/* ================= ONBOARDING PAGES ================= */

import Onboarding from "./pages/Onboarding";
import CloudProviderSetup from "./pages/CloudProviderSetup";

/* ========================================================= */
/* PROTECTED ROUTE */
/* ========================================================= */

function ProtectedRoute() {
  const isAuthenticated =
    localStorage.getItem("cloudmorph_auth") === "true";

  return isAuthenticated ? (
    <Outlet />
  ) : (
    <Navigate to="/login" replace />
  );
}

/* ========================================================= */
/* APP LAYOUT */
/* ========================================================= */

function AppLayout() {
  return (
    <div className="min-h-screen bg-slate-100">
      <Sidebar />

      <main className="ml-72 min-h-screen min-w-0">
        <Outlet />
      </main>
    </div>
  );
}

/* ========================================================= */
/* APP */
/* ========================================================= */

function App() {
  return (
    <AppProvider>
      <BrowserRouter>
        <Routes>
          {/* ================= PUBLIC ROUTES ================= */}

          <Route path="/" element={<Landing />} />

          <Route path="/login" element={<Login />} />

          <Route path="/signup" element={<Signup />} />

          <Route
            path="/forgot-password"
            element={<ForgotPassword />}
          />

          {/* ================= PROTECTED APP ================= */}

          <Route element={<ProtectedRoute />}>
            <Route path="/app" element={<AppLayout />}>
              {/* Dashboard */}

              <Route index element={<Dashboard />} />

              {/* ================= ONBOARDING ================= */}

              <Route
                path="onboarding"
                element={<Onboarding />}
              />

              <Route
                path="onboarding/cloud-provider"
                element={<CloudProviderSetup />}
              />

              {/* ================= PROFILE ================= */}

              <Route
                path="profile"
                element={<Profile />}
              />

              {/* ================= NOTIFICATIONS ================= */}

              <Route
                path="notifications"
                element={<Notifications />}
              />

              {/* ================= APPLICATIONS ================= */}

              <Route
                path="applications"
                element={<Applications />}
              />

              <Route
                path="applications/:id"
                element={<ApplicationDetails />}
              />

              {/* ================= AI ANALYSIS ================= */}

              <Route
                path="applications/:id/analysis"
                element={<AIAnalysis />}
              />

              <Route
                path="applications/:id/ai-analysis"
                element={<AIAnalysis />}
              />

              {/* ================= AI OPTIMIZER ================= */}

              <Route
                path="optimizer"
                element={<AIAnalysis />}
              />

              {/* ================= INFRASTRUCTURE ================= */}

              <Route
                path="infrastructure"
                element={<Infrastructure />}
              />

              {/* ================= ANALYTICS ================= */}

              <Route
                path="analytics"
                element={<Analytics />}
              />

              {/* ================= FORECASTING ================= */}

              <Route
                path="forecasting"
                element={<Forecasting />}
              />

              {/* ================= SETTINGS ================= */}

              <Route
                path="settings"
                element={<Settings />}
              />
            </Route>
          </Route>

          {/* ================= FALLBACK ================= */}

          <Route
            path="*"
            element={<Navigate to="/" replace />}
          />
        </Routes>
      </BrowserRouter>
    </AppProvider>
  );
}

export default App;
