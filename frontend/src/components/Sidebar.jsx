
import { NavLink, useNavigate } from "react-router-dom";

import {
  LayoutDashboard,
  AppWindow,
  BrainCircuit,
  Server,
  BarChart3,
  Activity,
  Settings,
  HelpCircle,
  LogOut,
  UserCircle2,
  Bell,
} from "lucide-react";

import cloudMorphLogo from "../assets/cloudmorph-logo.png";
import { useApp } from "../context/AppContext";

function Sidebar() {
  const navigate = useNavigate();
  const { user } = useApp();

  const handleLogout = () => {
    localStorage.removeItem("cloudmorph_auth");
    localStorage.removeItem("cloudmorph_user");
    localStorage.removeItem("cloudmorph_user_data");
    localStorage.removeItem("cloudmorph_applications");
    localStorage.removeItem("cloudmorph_onboarding");

    navigate("/login");
  };

  const navItems = [
    {
      name: "Dashboard",
      icon: LayoutDashboard,
      path: "/app",
    },
    {
      name: "Applications",
      icon: AppWindow,
      path: "/app/applications",
    },
    {
      name: "AI Optimizer",
      icon: BrainCircuit,
      path: "/app/optimizer",
    },
    {
      name: "Infrastructure",
      icon: Server,
      path: "/app/infrastructure",
    },
    {
      name: "Analytics",
      icon: BarChart3,
      path: "/app/analytics",
    },
    {
      name: "Forecasting",
      icon: Activity,
      path: "/app/forecasting",
    },
  ];

  return (
    <aside className="fixed left-0 top-0 z-40 flex h-screen w-72 flex-col border-r border-slate-800 bg-gradient-to-b from-slate-950 via-slate-950 to-slate-900 text-white">
      {/* ================= LOGO ================= */}

      <div className="relative border-b border-slate-800 px-6 py-5">
        <div className="absolute -top-6 right-4 h-28 w-28 rounded-full bg-blue-500/10 blur-3xl" />

        <NavLink
          to="/app"
          className="relative flex items-center gap-4 transition hover:opacity-95"
        >
          <div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-gradient-to-br from-slate-800 via-slate-900 to-slate-800 ring-1 ring-white/10 shadow-xl shadow-blue-900/40">
            <img
              src={cloudMorphLogo}
              alt="CloudMorph Logo"
              className="h-10 w-10 object-contain drop-shadow-lg"
            />
          </div>

          <div>
            <h1 className="text-[22px] font-extrabold tracking-tight text-white">
              CloudMorph
            </h1>

            <div className="mt-1 flex items-center gap-2">
              <span className="h-2 w-2 animate-pulse rounded-full bg-emerald-400" />

              <p className="text-xs font-medium text-slate-400">
                Adaptive Cloud Intelligence
              </p>
            </div>
          </div>
        </NavLink>
      </div>

      {/* ================= USER PROFILE ================= */}

      <div className="border-b border-slate-800 px-4 py-5">
        <NavLink
          to="/app/profile"
          className="group flex items-center gap-3 rounded-2xl border border-slate-700/50 bg-slate-800/50 p-3 transition duration-300 hover:border-blue-500/30 hover:bg-slate-800"
        >
          <div className="rounded-full bg-gradient-to-br from-blue-500 to-violet-600 p-2 shadow-lg shadow-blue-900/30">
            <UserCircle2 size={30} />
          </div>

          <div className="min-w-0 flex-1">
            <p className="truncate text-sm font-semibold text-white group-hover:text-blue-300">
              {user.name}
            </p>

            <p className="truncate text-xs text-slate-400">
              {user.email}
            </p>
          </div>
        </NavLink>
      </div>

      {/* ================= NAVIGATION ================= */}

      <nav className="flex-1 space-y-2 overflow-y-auto px-4 py-6">
        <p className="mb-3 px-3 text-xs font-bold uppercase tracking-[0.18em] text-slate-500">
          Workspace
        </p>

        {navItems.map((item) => {
          const Icon = item.icon;

          return (
            <NavLink
              key={item.name}
              to={item.path}
              end={item.path === "/app"}
              className={({ isActive }) =>
                `group relative flex items-center gap-3 rounded-xl px-4 py-3 text-sm font-medium transition-all duration-300 ${
                  isActive
                    ? "bg-gradient-to-r from-blue-600 to-violet-600 text-white shadow-lg shadow-blue-950/50"
                    : "text-slate-400 hover:bg-slate-800 hover:text-white"
                }`
              }
            >
              {({ isActive }) => (
                <>
                  {isActive && (
                    <span className="absolute -left-4 h-8 w-1 rounded-r-full bg-blue-400" />
                  )}

                  <Icon
                    size={20}
                    className={
                      isActive
                        ? "text-white"
                        : "text-slate-500 transition group-hover:text-blue-400"
                    }
                  />

                  <span>{item.name}</span>

                  {item.name === "AI Optimizer" && (
                    <span
                      className={`ml-auto rounded-full px-2 py-0.5 text-[10px] font-bold ${
                        isActive
                          ? "bg-white/15 text-blue-100"
                          : "bg-violet-500/15 text-violet-300"
                      }`}
                    >
                      AI
                    </span>
                  )}
                </>
              )}
            </NavLink>
          );
        })}

        <div className="my-6 border-t border-slate-800" />

        <p className="mb-3 px-3 text-xs font-bold uppercase tracking-[0.18em] text-slate-500">
          System
        </p>

        {/* Notifications */}

        <NavLink
          to="/app/notifications"
          className={({ isActive }) =>
            `group relative flex items-center gap-3 rounded-xl px-4 py-3 text-sm font-medium transition-all duration-300 ${
              isActive
                ? "bg-gradient-to-r from-blue-600 to-violet-600 text-white shadow-lg shadow-blue-950/50"
                : "text-slate-400 hover:bg-slate-800 hover:text-white"
            }`
          }
        >
          {({ isActive }) => (
            <>
              {isActive && (
                <span className="absolute -left-4 h-8 w-1 rounded-r-full bg-blue-400" />
              )}

              <Bell
                size={20}
                className={
                  isActive
                    ? "text-white"
                    : "text-slate-500 transition group-hover:text-blue-400"
                }
              />

              <span>Notifications</span>
            </>
          )}
        </NavLink>

        {/* Settings */}

        <NavLink
          to="/app/settings"
          className={({ isActive }) =>
            `group relative flex items-center gap-3 rounded-xl px-4 py-3 text-sm font-medium transition-all duration-300 ${
              isActive
                ? "bg-gradient-to-r from-blue-600 to-violet-600 text-white shadow-lg shadow-blue-950/50"
                : "text-slate-400 hover:bg-slate-800 hover:text-white"
            }`
          }
        >
          {({ isActive }) => (
            <>
              {isActive && (
                <span className="absolute -left-4 h-8 w-1 rounded-r-full bg-blue-400" />
              )}

              <Settings
                size={20}
                className={
                  isActive
                    ? "text-white"
                    : "text-slate-500 transition group-hover:text-blue-400"
                }
              />

              <span>Settings</span>
            </>
          )}
        </NavLink>
      </nav>

      {/* ================= BOTTOM PANEL ================= */}

      <div className="border-t border-slate-800 p-4">
        {/* AI ENGINE */}

        <div className="relative overflow-hidden rounded-2xl border border-blue-500/20 bg-gradient-to-br from-blue-600/20 via-slate-900 to-violet-600/20 p-4">
          <div className="absolute -right-6 -top-6 h-20 w-20 rounded-full bg-blue-500/20 blur-2xl" />

          <div className="relative flex items-center gap-3">
            <div className="rounded-xl bg-blue-500/20 p-2.5 text-blue-300">
              <BrainCircuit size={20} />
            </div>

            <div>
              <p className="text-sm font-semibold text-white">
                AI Engine
              </p>

              <div className="mt-1 flex items-center gap-1.5">
                <span className="h-2 w-2 animate-pulse rounded-full bg-emerald-400" />

                <span className="text-xs text-slate-400">
                  Active & Monitoring
                </span>
              </div>
            </div>
          </div>

          <button className="relative mt-4 flex w-full items-center justify-center gap-2 rounded-xl border border-white/10 bg-white/5 py-2.5 text-xs font-semibold text-slate-300 transition hover:bg-white/10 hover:text-white">
            <HelpCircle size={15} />
            AI Assistant
          </button>
        </div>

        {/* LOGOUT */}

        <button
          onClick={handleLogout}
          className="mt-4 flex w-full items-center justify-center gap-2 rounded-xl border border-red-500/20 bg-red-500/10 py-3 text-sm font-semibold text-red-300 transition hover:bg-red-500/20 hover:text-red-200"
        >
          <LogOut size={18} />
          Logout
        </button>
      </div>
    </aside>
  );
}

export default Sidebar;
