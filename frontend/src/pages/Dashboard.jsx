
import { useCallback, useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";

import {
  DollarSign,
  TrendingDown,
  Gauge,
  Layers3,
  ArrowRight,
  Sparkles,
  BrainCircuit,
  Cloud,
  Server,
  Activity,
  Plus,
  BarChart3,
} from "lucide-react";

import { useApp } from "../context/AppContext";
import API from "../services/api";
import cloudMorphLogo from "../assets/cloudmorph-logo.png";
import AddApplicationModal from "../components/AddApplicationModal";

/* ========================================================= */
/* RESOLVE THE LOGGED-IN USER'S DISPLAY NAME */
/* ========================================================= */

function getDisplayName(contextUser) {
  const isPlaceholder = (value) =>
    !value ||
    ["user", "cloudmorph user"].includes(
      String(value).trim().toLowerCase()
    );

  let savedUser = {};

  try {
    savedUser = JSON.parse(
      localStorage.getItem("cloudmorph_user_data") || "{}"
    );
  } catch {
    savedUser = {};
  }

  const candidates = [
    contextUser?.name,
    contextUser?.full_name,
    savedUser?.name,
    savedUser?.full_name,
    savedUser?.fullName,
    savedUser?.username,
    savedUser?.first_name,
    savedUser?.firstName,
  ];

  const actualName = candidates.find(
    (value) =>
      typeof value === "string" &&
      !isPlaceholder(value) &&
      !value.includes("@")
  );

  if (actualName) {
    return actualName.trim();
  }

  const email =
    contextUser?.email ||
    savedUser?.email ||
    localStorage.getItem("cloudmorph_email") ||
    localStorage.getItem("cloudmorph_user") ||
    "";

  if (email.includes("@")) {
    const emailName = email
      .split("@")[0]
      .replace(/[._-]+/g, " ")
      .trim();

    if (emailName && !isPlaceholder(emailName)) {
      return emailName.replace(/\b\w/g, (char) =>
        char.toUpperCase()
      );
    }
  }

  return "there";
}

/* ========================================================= */
/* NORMALIZE BACKEND APPLICATION DATA */
/* ========================================================= */

function toFiniteNumber(value, fallback = 0) {
  const number = Number(value);

  return Number.isFinite(number) ? number : fallback;
}

function normalizeApplication(app) {
  const cost = Math.max(
    0,
    toFiniteNumber(
      app.cost ?? app.monthly_cost ?? app.monthlyCost
    )
  );

  const rawOptimizedCost = toFiniteNumber(
    app.optimized_cost ??
      app.optimizedCost ??
      app.optimized_cost_monthly,
    0
  );

  // Do not invent savings when no valid optimization estimate exists.
  // An optimized cost cannot produce negative savings in the dashboard.
  const optimizedCost =
    rawOptimizedCost > 0
      ? Math.min(rawOptimizedCost, cost)
      : cost;

  const utilization = Math.min(
    100,
    Math.max(
      0,
      toFiniteNumber(app.utilization ?? app.usagePercent)
    )
  );

  const decision =
    typeof app.decision === "string" && app.decision.trim()
      ? app.decision
      : "ANALYZE";

  return {
    ...app,
    id: String(app.id ?? app._id ?? ""),
    name: app.name || "Unnamed Application",
    provider:
      typeof app.provider === "string" && app.provider.trim()
        ? app.provider.trim()
        : "AWS",
    cost,
    optimizedCost,
    utilization,
    decision,
    status: app.status || "Active",
  };
}

/* ========================================================= */
/* DASHBOARD */
/* ========================================================= */

function Dashboard() {
  const { user } = useApp();

  const displayName = getDisplayName(user);

  const [applications, setApplications] = useState([]);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  /* ========================================================= */
  /* FETCH APPLICATIONS FROM BACKEND */
  /* ========================================================= */

  const fetchApplications = useCallback(async () => {
    setLoading(true);
    setError("");

    try {
      const response = await API.get("/applications");
      const responseData = response.data;

      let data = [];

      if (Array.isArray(responseData)) {
        data = responseData;
      } else if (Array.isArray(responseData?.applications)) {
        data = responseData.applications;
      } else if (Array.isArray(responseData?.data)) {
        data = responseData.data;
      } else if (
        responseData &&
        typeof responseData === "object" &&
        Object.keys(responseData).length === 0
      ) {
        data = [];
      } else {
        throw new Error(
          "The backend returned an unexpected applications response."
        );
      }

      const normalizedApplications = data
        .filter(
          (app) => app && typeof app === "object" && !Array.isArray(app)
        )
        .map(normalizeApplication);

      setApplications(normalizedApplications);
    } catch (err) {
      console.error("Failed to fetch dashboard applications:", err);

      setApplications([]);

      setError(
        err.response?.data?.detail ||
          err.message ||
          "Unable to load applications. Please check your connection and try again."
      );
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchApplications();
  }, [fetchApplications]);

  /* ========================================================= */
  /* DASHBOARD CALCULATIONS */
  /* ========================================================= */

  const stats = useMemo(() => {
    const totalCost = applications.reduce(
      (sum, app) => sum + app.cost,
      0
    );

    // Calculate savings only from saved optimization estimates.
    // Missing estimates produce zero savings instead of invented savings.
    const totalSavings = applications.reduce(
      (sum, app) =>
        sum + Math.max(0, app.cost - app.optimizedCost),
      0
    );

    const optimizedCost = totalCost - totalSavings;

    const utilization =
      applications.length > 0
        ? Math.round(
            applications.reduce(
              (sum, app) => sum + app.utilization,
              0
            ) / applications.length
          )
        : 0;

    const savingsPercentage =
      totalCost > 0
        ? Math.round((totalSavings / totalCost) * 100)
        : 0;

    return {
      totalCost,
      optimizedCost,
      totalSavings,
      utilization,
      savingsPercentage,
      applicationCount: applications.length,
    };
  }, [applications]);

  const providerDistribution = useMemo(() => {
    const aws = applications.filter(
      (app) => app.provider.trim().toUpperCase() === "AWS"
    ).length;

    const gcp = applications.filter(
      (app) => app.provider.trim().toUpperCase() === "GCP"
    ).length;

    const azure = applications.filter(
      (app) => app.provider.trim().toUpperCase() === "AZURE"
    ).length;

    return { aws, gcp, azure };
  }, [applications]);

  /* ========================================================= */
  /* RENDER */
  /* ========================================================= */

  return (
    <div className="min-h-screen bg-slate-100">
      {/* HERO */}

      <section className="relative overflow-hidden bg-gradient-to-br from-slate-950 via-blue-950 to-violet-950">
        <div className="absolute -left-24 top-0 h-72 w-72 rounded-full bg-blue-500/20 blur-3xl" />
        <div className="absolute -right-24 bottom-0 h-80 w-80 rounded-full bg-violet-500/20 blur-3xl" />

        <div className="relative px-8 py-10">
          <div className="flex flex-col gap-8 lg:flex-row lg:items-center lg:justify-between">
            <div>
              <div className="flex items-center gap-3">
                <div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-gradient-to-br from-slate-800 via-slate-900 to-slate-800 shadow-xl ring-1 ring-white/10">
                  <img
                    src={cloudMorphLogo}
                    alt="CloudMorph"
                    className="h-10 w-10 object-contain"
                  />
                </div>

                <div>
                  <h1 className="text-3xl font-bold text-white md:text-5xl">
                    Welcome back, {displayName}
                  </h1>

                  <p className="mt-1 text-sm text-slate-300">
                    Review your cloud application costs and saved optimization insights
                    with CloudMorph.
                  </p>
                </div>
              </div>

              <div className="mt-6 flex flex-wrap gap-3">
                <span className="rounded-full border border-blue-400/20 bg-blue-500/10 px-4 py-2 text-xs font-semibold text-blue-200">
                  Optimization Insights Available
                </span>

                <span className="rounded-full border border-emerald-400/20 bg-emerald-500/10 px-4 py-2 text-xs font-semibold text-emerald-200">
                  Application Data Connected
                </span>
              </div>
            </div>

            <button
              type="button"
              onClick={() => setIsModalOpen(true)}
              className="inline-flex items-center gap-2 rounded-xl bg-gradient-to-r from-blue-600 to-violet-600 px-5 py-3 font-semibold text-white transition hover:scale-[1.02]"
            >
              <Plus size={18} />
              Add Application
            </button>
          </div>

          {/* HERO METRICS */}

          <div className="mt-10 grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-4">
            <HeroCard
              icon={DollarSign}
              label="Current Cost"
              value={`$${stats.totalCost.toFixed(2)}`}
            />

            <HeroCard
              icon={TrendingDown}
              label="Potential Savings"
              value={`$${stats.totalSavings.toFixed(2)}`}
            />

            <HeroCard
              icon={Gauge}
              label="Utilization"
              value={`${stats.utilization}%`}
            />

            <HeroCard
              icon={Layers3}
              label="Applications"
              value={stats.applicationCount}
            />
          </div>
        </div>
      </section>

      {/* MAIN */}

      <main className="space-y-8 p-8">
        {/* LOADING AND ERROR STATES */}

        {loading && (
          <div
            role="status"
            className="rounded-2xl border border-blue-100 bg-white px-5 py-3 text-sm text-blue-700 shadow-sm"
          >
            Loading your applications...
          </div>
        )}

        {!loading && error && (
          <div
            role="alert"
            className="flex flex-wrap items-center justify-between gap-3 rounded-2xl border border-red-200 bg-white px-5 py-4 text-sm text-red-700 shadow-sm"
          >
            <span>{error}</span>

            <button
              type="button"
              onClick={fetchApplications}
              className="rounded-lg bg-red-50 px-4 py-2 font-semibold transition hover:bg-red-100"
            >
              Try Again
            </button>
          </div>
        )}

        {/* QUICK ACTIONS */}

        <section className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-4">
          <QuickAction
            icon={BrainCircuit}
            title="AI Optimizer"
            description="View optimization recommendations"
            to="/app/optimizer"
          />

          <QuickAction
            icon={BarChart3}
            title="Analytics"
            description="Cost & performance insights"
            to="/app/analytics"
          />

          <QuickAction
            icon={Server}
            title="Infrastructure"
            description="Monitor cloud resources"
            to="/app/infrastructure"
          />

          <QuickAction
            icon={Cloud}
            title="Applications"
            description="Manage cloud workloads"
            to="/app/applications"
          />
        </section>

        {/* COST SUMMARY */}

        <section className="grid grid-cols-1 gap-6 xl:grid-cols-3">
          <div className="rounded-3xl border border-slate-200 bg-white p-7 shadow-sm xl:col-span-2">
            <div className="flex items-center gap-3">
              <TrendingDown className="text-blue-600" size={24} />

              <div>
                <h2 className="text-xl font-bold text-slate-800">
                  Cost Optimization Overview
                </h2>

                <p className="text-sm text-slate-500">
                  Review current costs and saved optimization estimates.
                </p>
              </div>
            </div>

            <div className="mt-8 space-y-6">
              <ProgressRow
                label="Current Cost"
                value={`$${stats.totalCost.toFixed(2)}`}
                percentage={100}
                color="bg-blue-500"
              />

              <ProgressRow
                label="Optimized Cost"
                value={`$${stats.optimizedCost.toFixed(2)}`}
                percentage={
                  stats.totalCost > 0
                    ? Math.min(
                        100,
                        (stats.optimizedCost / stats.totalCost) * 100
                      )
                    : 0
                }
                color="bg-emerald-500"
              />

              <ProgressRow
                label="Estimated Savings"
                value={`${stats.savingsPercentage}%`}
                percentage={Math.min(
                  100,
                  Math.max(0, stats.savingsPercentage)
                )}
                color="bg-violet-500"
              />
            </div>
          </div>

          {/* AI STATUS */}

          <div className="rounded-3xl border border-slate-200 bg-white p-7 shadow-sm">
            <div className="flex items-center gap-3">
              <Sparkles className="text-violet-600" size={24} />

              <div>
                <h2 className="text-xl font-bold text-slate-800">
                  AI Status
                </h2>

                <p className="text-sm text-slate-500">
                  Optimization engine health
                </p>
              </div>
            </div>

            <div className="mt-8 flex items-center justify-center">
              <div className="relative flex h-40 w-40 items-center justify-center rounded-full bg-gradient-to-br from-violet-100 via-blue-50 to-emerald-100">
                <div className="absolute inset-3 flex items-center justify-center rounded-full bg-white shadow-inner">
                  <div className="text-center">
                    <p className="text-4xl font-bold text-slate-800">
                      {stats.savingsPercentage}%
                    </p>

                    <p className="text-xs text-slate-400">
                      Optimization
                    </p>
                  </div>
                </div>
              </div>
            </div>

            <div className="mt-8 rounded-2xl bg-slate-50 p-4">
              <div className="flex items-center gap-3">
                <Activity
                  className="text-emerald-500"
                  size={20}
                />

                <div>
                  <p className="text-sm font-semibold text-slate-800">
                    Optimization Engine Available
                  </p>

                  <p className="text-xs text-slate-500">
                    Using saved application metrics and cost estimates
                  </p>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* APPLICATIONS */}

        <section className="rounded-3xl border border-slate-200 bg-white p-7 shadow-sm">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-xl font-bold text-slate-800">
                Connected Applications
              </h2>

              <p className="mt-1 text-sm text-slate-500">
                Live cloud applications connected to CloudMorph.
              </p>
            </div>

            <Link
              to="/app/applications"
              className="flex items-center gap-2 text-sm font-semibold text-blue-600 hover:text-blue-700"
            >
              View All
              <ArrowRight size={16} />
            </Link>
          </div>

          <div className="mt-7 grid grid-cols-1 gap-5 xl:grid-cols-2">
            {!loading && applications.length === 0 && !error && (
              <div className="rounded-2xl border border-dashed border-slate-300 p-8 text-center xl:col-span-2">
                <Cloud className="mx-auto text-slate-400" size={32} />

                <h3 className="mt-3 font-bold text-slate-800">
                  No applications yet
                </h3>

                <p className="mt-1 text-sm text-slate-500">
                  Add an application to see its cloud metrics here.
                </p>

                <button
                  type="button"
                  onClick={() => setIsModalOpen(true)}
                  className="mt-4 inline-flex items-center gap-2 rounded-xl bg-blue-600 px-4 py-2 font-semibold text-white transition hover:bg-blue-700"
                >
                  <Plus size={16} />
                  Add Application
                </button>
              </div>
            )}

            {applications.slice(0, 4).map((app) => (
              <Link
                key={app.id}
                to={`/app/applications/${encodeURIComponent(app.id)}`}
                className="group rounded-2xl border border-slate-200 bg-slate-50 p-5 transition hover:-translate-y-1 hover:border-blue-200 hover:bg-white hover:shadow-lg"
              >
                <div className="flex items-center justify-between">
                  <div>
                    <h3 className="text-lg font-bold text-slate-800 group-hover:text-blue-600">
                      {app.name}
                    </h3>

                    <p className="mt-1 text-sm text-slate-500">
                      {app.provider}
                    </p>
                  </div>

                  <span className="rounded-lg bg-blue-50 px-3 py-1 text-xs font-bold text-blue-600">
                    {app.decision}
                  </span>
                </div>

                <div className="mt-5 grid grid-cols-3 gap-3">
                  <SmallStat
                    label="Cost"
                    value={`$${app.cost.toFixed(2)}`}
                  />

                  <SmallStat
                    label="Optimized"
                    value={`$${app.optimizedCost.toFixed(2)}`}
                  />

                  <SmallStat
                    label="Utilization"
                    value={`${app.utilization}%`}
                  />
                </div>
              </Link>
            ))}
          </div>
        </section>

        {/* PROVIDER DISTRIBUTION */}

        <section className="grid grid-cols-1 gap-6 xl:grid-cols-2">
          <div className="rounded-3xl border border-slate-200 bg-white p-7 shadow-sm">
            <h2 className="text-xl font-bold text-slate-800">
              Cloud Provider Distribution
            </h2>

            <p className="mt-1 text-sm text-slate-500">
              Connected cloud providers.
            </p>

            <div className="mt-7 space-y-5">
              <ProviderRow
                name="AWS"
                count={providerDistribution.aws}
                total={stats.applicationCount}
                color="bg-orange-500"
              />

              <ProviderRow
                name="GCP"
                count={providerDistribution.gcp}
                total={stats.applicationCount}
                color="bg-blue-500"
              />

              <ProviderRow
                name="Azure"
                count={providerDistribution.azure}
                total={stats.applicationCount}
                color="bg-cyan-500"
              />
            </div>
          </div>

          <div className="rounded-3xl bg-gradient-to-br from-blue-600 to-violet-700 p-7 text-white shadow-sm">
            <Sparkles size={30} />

            <h2 className="mt-5 text-2xl font-bold">
              CloudMorph AI Insight
            </h2>

            <p className="mt-3 text-blue-100">
              Your infrastructure currently has an estimated savings
              opportunity of{" "}
              <span className="font-bold text-white">
                {stats.savingsPercentage}%
              </span>{" "}
              based on the saved optimization cost estimates.
            </p>

            <div className="mt-8 rounded-2xl bg-white/10 p-4 backdrop-blur">
              <div className="flex items-center justify-between">
                <span className="text-sm">Potential Savings</span>

                <span className="text-xl font-bold">
                  ${stats.totalSavings.toFixed(2)}
                </span>
              </div>
            </div>
          </div>
        </section>

        <div className="h-2" />
      </main>

      {/* ADD APPLICATION MODAL */}

      <AddApplicationModal
        isOpen={isModalOpen}
        onClose={() => {
          setIsModalOpen(false);
          fetchApplications();
        }}
      />
    </div>
  );
}

/* ========================================================= */
/* COMPONENTS */
/* ========================================================= */

function HeroCard({ icon: Icon, label, value }) {
  return (
    <div className="rounded-2xl border border-white/10 bg-white/5 p-5 backdrop-blur transition hover:bg-white/10">
      <div className="flex items-center gap-3">
        <div className="rounded-xl bg-white/10 p-3 text-blue-200">
          <Icon size={22} />
        </div>

        <div>
          <p className="text-xs text-slate-400">{label}</p>

          <p className="mt-1 text-2xl font-bold text-white">
            {value}
          </p>
        </div>
      </div>
    </div>
  );
}

function QuickAction({ icon: Icon, title, description, to }) {
  return (
    <Link
      to={to}
      className="group rounded-2xl border border-slate-200 bg-white p-5 shadow-sm transition hover:-translate-y-1 hover:border-blue-200 hover:shadow-lg"
    >
      <div className="flex items-center justify-between">
        <div className="rounded-xl bg-blue-50 p-3 text-blue-600">
          <Icon size={22} />
        </div>

        <ArrowRight
          size={18}
          className="text-slate-300 transition group-hover:text-blue-600"
        />
      </div>

      <h3 className="mt-5 text-lg font-bold text-slate-800">
        {title}
      </h3>

      <p className="mt-1 text-sm text-slate-500">
        {description}
      </p>
    </Link>
  );
}

function ProgressRow({ label, value, percentage, color }) {
  const safePercentage = Math.min(
    100,
    Math.max(0, Number(percentage) || 0)
  );

  return (
    <div>
      <div className="flex items-center justify-between">
        <span className="text-sm font-semibold text-slate-700">
          {label}
        </span>

        <span className="text-sm font-bold text-slate-800">
          {value}
        </span>
      </div>

      <div className="mt-2 h-2 overflow-hidden rounded-full bg-slate-100">
        <div
          className={`h-full rounded-full ${color}`}
          style={{ width: `${safePercentage}%` }}
        />
      </div>
    </div>
  );
}

function SmallStat({ label, value }) {
  return (
    <div className="rounded-xl bg-white p-3">
      <p className="text-[10px] text-slate-400">{label}</p>

      <p className="mt-1 text-sm font-bold text-slate-800">
        {value}
      </p>
    </div>
  );
}

function ProviderRow({ name, count, total, color }) {
  const percentage =
    total > 0 ? Math.round((count / total) * 100) : 0;

  return (
    <div>
      <div className="flex items-center justify-between">
        <span className="font-semibold text-slate-700">
          {name}
        </span>

        <span className="text-sm text-slate-500">
          {count} Apps
        </span>
      </div>

      <div className="mt-2 h-2 overflow-hidden rounded-full bg-slate-100">
        <div
          className={`h-full rounded-full ${color}`}
          style={{ width: `${percentage}%` }}
        />
      </div>
    </div>
  );
}

export default Dashboard;
