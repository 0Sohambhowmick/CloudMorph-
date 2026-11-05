import { useCallback, useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import API from "../services/api";
import {
  Plus,
  Search,
  MoreVertical,
  Server,
  Cloud,
  Activity,
  DollarSign,
  CheckCircle2,
  AlertTriangle,
  ChevronRight,
  ChevronDown,
  TrendingUp,
  Sparkles,
  BrainCircuit,
  ArrowUpRight,
  RefreshCw,
  X,
  Layers3,
  CircleDollarSign,
  ShieldCheck,
  SlidersHorizontal,
  BarChart3,
  Cpu,
} from "lucide-react";

const toFiniteNumber = (value, fallback = 0) => {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
};

const normalizeApplication = (app) => {
  const cost = Math.max(
    0,
    toFiniteNumber(app.monthlyCost ?? app.monthly_cost ?? app.cost, 0)
  );
  const optimizedCost = Math.max(
    0,
    toFiniteNumber(
      app.optimized_cost ?? app.optimizedCost,
      cost
    )
  );
  const utilizationValue =
    app.usagePercent ??
    app.usage_percent ??
    app.utilization ??
    app.utilization_percent;
  const parsedUtilization =
    utilizationValue === null || utilizationValue === undefined || utilizationValue === ""
      ? null
      : Number(utilizationValue);
  const rawUtilization =
    parsedUtilization !== null && Number.isFinite(parsedUtilization)
      ? parsedUtilization
      : null;
  const usagePercent =
    rawUtilization === null
      ? null
      : rawUtilization > 0 && rawUtilization <= 1
        ? Math.round(rawUtilization * 100)
        : Math.max(0, Math.min(100, rawUtilization));
  // Prefer the latest persisted optimizer result, while retaining compatibility
  // with application records created before latest_optimization was added.
  const latestOptimization =
    app.latest_optimization && typeof app.latest_optimization === "object"
      ? app.latest_optimization
      : app.latestOptimization && typeof app.latestOptimization === "object"
        ? app.latestOptimization
        : null;
  const latestMlAnalysis =
    latestOptimization?.ml_analysis ??
    latestOptimization?.mlAnalysis ??
    null;
  const latestPrediction =
    latestMlAnalysis?.prediction ??
    latestMlAnalysis?.result ??
    null;

  const decision = String(
    latestOptimization?.action ??
    latestOptimization?.decision ??
    app.decision ??
    ""
  ).toUpperCase();

  const workloadPrediction = String(
    latestPrediction?.predicted_condition ??
    latestPrediction?.prediction ??
    latestMlAnalysis?.predicted_condition ??
    ""
  ).toUpperCase();

  const confidenceValue =
    latestPrediction?.confidence ??
    latestPrediction?.probability ??
    latestMlAnalysis?.confidence ??
    null;
  const confidenceNumber =
    confidenceValue === null || confidenceValue === undefined || confidenceValue === ""
      ? null
      : Number(confidenceValue);
  const workloadConfidence =
    confidenceNumber !== null && Number.isFinite(confidenceNumber)
      ? Math.max(0, Math.min(100, confidenceNumber <= 1 ? confidenceNumber * 100 : confidenceNumber))
      : null;

  const decisionSource = String(
    latestOptimization?.decision_source ??
    latestOptimization?.decisionSource ??
    ""
  );

  const rawStatus = String(app.status ?? "").toLowerCase();
  const status =
    rawStatus === "warning" || rawStatus === "critical" || rawStatus === "unhealthy"
      ? "Warning"
      : rawStatus === "healthy" || rawStatus === "active"
        ? "Healthy"
        : decision === "SWITCH" || decision === "RESIZE" || (usagePercent !== null && usagePercent >= 80)
          ? "Warning"
          : "Healthy";
  const rawProvider = String(app.provider ?? "AWS");
  const providerLower = rawProvider.toLowerCase();
  const provider =
    providerLower === "google cloud" || providerLower === "google cloud platform"
      ? "GCP"
      : providerLower === "microsoft azure"
        ? "Azure"
        : providerLower === "amazon web services"
          ? "AWS"
          : rawProvider;
  const rawSavings =
    app.potentialSavings ?? app.potential_savings ?? app.savings;
  const savings =
    rawSavings !== undefined && rawSavings !== null
      ? Math.max(0, toFiniteNumber(rawSavings, 0))
      : Math.max(0, Math.round((cost - optimizedCost) * 100) / 100);
  const rawServices = toFiniteNumber(app.services ?? app.service_count, 1);
  const rawHealth = toFiniteNumber(
    app.healthScore ?? app.health_score,
    status === "Healthy" ? 90 : 70
  );

  return {
    id: String(app.id ?? app._id ?? app.application_id ?? ""),
    name: String(app.name ?? "Unnamed Application"),
    description: String(app.description ?? app.service ?? "Cloud application"),
    provider,
    providerShort: provider.toLowerCase(),
    services: Math.max(0, Math.floor(rawServices)),
    monthlyCost: cost,
    status,
    usage:
      app.usage ||
      (usagePercent === null
        ? "Unknown"
        : usagePercent >= 85
          ? "High"
          : usagePercent >= 50
            ? "Medium"
            : "Low"),
    usagePercent,
    healthScore: Math.max(0, Math.min(100, rawHealth)),
    optimizationScore: Math.max(0, Math.min(100, toFiniteNumber(app.optimizationScore, 80))),
    decision,
    recommendation: String(
      latestOptimization?.title ??
      latestOptimization?.recommendation ??
      app.recommendation ??
      app.ai_recommendation ??
      `CloudMorph recommendation: ${
        usagePercent === null
          ? "INVESTIGATE"
          : usagePercent <= 35
            ? "RESIZE"
            : usagePercent >= 85
              ? "SCALE_UP"
              : "KEEP"
      }`
    ),
    hasLatestOptimization: Boolean(latestOptimization),
    latestOptimizationAction: decision || null,
    workloadPrediction: workloadPrediction || null,
    workloadConfidence,
    decisionSource: decisionSource || null,
    potentialSavings: savings,
    category: app.category || app.service || "Cloud Application",
  };
};

const getApplicationsFromResponse = (data) => {
  if (Array.isArray(data)) return data;
  if (Array.isArray(data?.applications)) return data.applications;
  if (Array.isArray(data?.items)) return data.items;
  if (Array.isArray(data?.data)) return data.data;
  return [];
};

function Applications() {
  const navigate = useNavigate();
  const [applications, setApplications] = useState([]);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [searchTerm, setSearchTerm] = useState("");
  const [providerFilter, setProviderFilter] = useState("All");
  const [statusFilter, setStatusFilter] = useState("All");
  const [sortBy, setSortBy] = useState("Cost High to Low");
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [isLoadingApplications, setIsLoadingApplications] = useState(true);
  const [pageError, setPageError] = useState("");
  const [activeMenuId, setActiveMenuId] = useState(null);
  const [deletingApplicationId, setDeletingApplicationId] = useState(null);

  const loadApplications = useCallback(async ({ showLoader = false } = {}) => {
    if (showLoader) setIsLoadingApplications(true);
    setPageError("");
    try {
      const response = await API.get("/applications");
      const records = getApplicationsFromResponse(response.data);
      setApplications(
        records
          .filter((record) => record && typeof record === "object")
          .map(normalizeApplication)
      );
    } catch (err) {
      if (err.response?.status === 401) {
        setPageError("Your session has expired. Please log in again.");
      } else if (err.response?.data?.detail) {
        setPageError(err.response.data.detail);
      } else if (err.response) {
        setPageError("Unable to load applications. Please try again.");
      } else {
        setPageError(
          "Unable to connect to CloudMorph API. Check that the backend is running."
        );
      }
    } finally {
      setIsLoadingApplications(false);
      setIsRefreshing(false);
    }
  }, []);

  useEffect(() => {
    loadApplications({ showLoader: true });
  }, [loadApplications]);

  const totalCost = applications.reduce(
    (total, app) => total + app.monthlyCost,
    0
  );
  const healthyApps = applications.filter(
    (app) => app.status === "Healthy"
  ).length;
  const warningApps = applications.filter(
    (app) => app.status === "Warning"
  ).length;
  const totalSavings = applications.reduce(
    (total, app) => total + app.potentialSavings,
    0
  );
  const totalServices = applications.reduce(
    (total, app) => total + app.services,
    0
  );

  const filteredApplications = useMemo(() => {
    let result = [...applications];
    if (searchTerm.trim()) {
      result = result.filter((app) =>
        `${app.name} ${app.description} ${app.provider} ${app.category}`
          .toLowerCase()
          .includes(searchTerm.toLowerCase())
      );
    }
    if (providerFilter !== "All") {
      result = result.filter((app) => app.provider === providerFilter);
    }
    if (statusFilter !== "All") {
      result = result.filter((app) => app.status === statusFilter);
    }
    if (sortBy === "Cost High to Low") {
      result.sort((a, b) => b.monthlyCost - a.monthlyCost);
    }
    if (sortBy === "Cost Low to High") {
      result.sort((a, b) => a.monthlyCost - b.monthlyCost);
    }
    if (sortBy === "Name A-Z") {
      result.sort((a, b) => a.name.localeCompare(b.name));
    }
    if (sortBy === "Health Score") {
      result.sort((a, b) => b.healthScore - a.healthScore);
    }
    return result;
  }, [applications, searchTerm, providerFilter, statusFilter, sortBy]);

  const clearFilters = () => {
    setSearchTerm("");
    setProviderFilter("All");
    setStatusFilter("All");
    setSortBy("Cost High to Low");
  };

  const handleRefresh = async () => {
    if (isRefreshing) return;
    setIsRefreshing(true);
    await loadApplications();
  };

  const handleDeleteApplication = async (application) => {
    if (!application?.id || deletingApplicationId) return;
    const confirmed = window.confirm(
      `Delete "${application.name}"? This action cannot be undone.`
    );
    if (!confirmed) {
      setActiveMenuId(null);
      return;
    }

    setDeletingApplicationId(application.id);
    setPageError("");
    try {
      await API.delete(`/applications/${encodeURIComponent(application.id)}`);
      setApplications((previous) =>
        previous.filter((item) => item.id !== application.id)
      );
      setActiveMenuId(null);
    } catch (err) {
      if (err.response?.status === 401) {
        setPageError("Your session has expired. Please log in again.");
      } else if (err.response?.data?.detail) {
        setPageError(err.response.data.detail);
      } else if (err.response) {
        setPageError("Unable to delete the application. Please try again.");
      } else {
        setPageError(
          "Unable to connect to CloudMorph API. Check that the backend is running."
        );
      }
    } finally {
      setDeletingApplicationId(null);
    }
  };

  const handleAddApplication = async (newApplication) => {
    const cost = Number(newApplication.monthlyCost);
    const payload = {
      name: newApplication.name.trim(),
      description: newApplication.description.trim(),
      provider: newApplication.provider,
      service: newApplication.category,
      category: newApplication.category,
      services: Number(newApplication.services),
      cost,
      utilization:
        newApplication.utilization === ""
          ? null
          : Number(newApplication.utilization),
      optimized_cost: cost,
      decision: "KEEP",
      status: newApplication.status === "Warning" ? "PENDING" : "ACTIVE",
      region: "",
    };
    try {
      setPageError("");
      const response = await API.post("/applications", payload);
      const savedApplication =
        response.data?.application ||
        response.data?.data ||
        response.data;
      if (savedApplication && (savedApplication.id || savedApplication._id)) {
        setApplications((previous) => [
          normalizeApplication(savedApplication),
          ...previous,
        ]);
      } else {
        await loadApplications();
      }
      setIsModalOpen(false);
      return true;
    } catch (err) {
      if (err.response?.status === 401) {
        setPageError("Your session has expired. Please log in again.");
      } else if (err.response?.data?.detail) {
        setPageError(err.response.data.detail);
      } else if (err.response) {
        setPageError("Unable to save the application. Please try again.");
      } else {
        setPageError(
          "Unable to connect to CloudMorph API. Check that the backend is running."
        );
      }
      return false;
    }
  };

  return (
    <div className="min-h-screen bg-slate-100">
      {/* ===================================================== */}
      {/* HERO HEADER */}
      {/* ===================================================== */}
      <div className="relative overflow-hidden border-b border-slate-200 bg-gradient-to-r from-slate-950 via-blue-950 to-violet-950 px-8 py-8">
        <div className="absolute -right-20 -top-20 h-72 w-72 rounded-full bg-blue-500/20 blur-3xl" />
        <div className="absolute bottom-0 left-1/3 h-40 w-40 rounded-full bg-violet-500/20 blur-3xl" />
        <div className="relative flex flex-col gap-6 xl:flex-row xl:items-center xl:justify-between">
          <div>
            <div className="flex flex-wrap items-center gap-3">
              <div className="flex items-center gap-2 rounded-full border border-blue-400/20 bg-blue-500/15 px-3 py-1">
                <Layers3 size={14} className="text-blue-300" />
                <span className="text-xs font-semibold text-blue-200">
                  CLOUD WORKSPACE
                </span>
              </div>
              <span className="flex items-center gap-1.5 text-xs text-emerald-300">
                <span className="h-2 w-2 animate-pulse rounded-full bg-emerald-400" />
                {applications.length} applications monitored
              </span>
            </div>
            <h1 className="mt-5 text-3xl font-bold tracking-tight text-white md:text-4xl">
              Cloud Applications
            </h1>
            <p className="mt-3 max-w-2xl text-sm leading-relaxed text-slate-300">
              Manage your applications, monitor cloud services, track costs,
              and discover AI-powered optimization opportunities.
            </p>
          </div>
          <div className="flex flex-wrap gap-3">
            <button
              type="button"
              onClick={handleRefresh}
              disabled={isRefreshing}
              className="flex items-center gap-2 rounded-xl border border-white/15 bg-white/10 px-4 py-3 text-sm font-semibold text-white backdrop-blur transition hover:bg-white/15 disabled:cursor-not-allowed disabled:opacity-70"
            >
              <RefreshCw
                size={17}
                className={isRefreshing ? "animate-spin" : ""}
              />
              {isRefreshing ? "Refreshing..." : "Refresh Data"}
            </button>
            <button
              type="button"
              onClick={() => setIsModalOpen(true)}
              className="flex items-center gap-2 rounded-xl bg-white px-5 py-3 text-sm font-bold text-slate-900 shadow-lg transition hover:bg-slate-100"
            >
              <Plus size={18} />
              Add Application
            </button>
          </div>
        </div>
      </div>

      {/* ===================================================== */}
      {/* MAIN CONTENT */}
      {/* ===================================================== */}
      <div className="p-8">
        {pageError && (
          <div
            role="alert"
            aria-live="polite"
            className="mb-6 rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700"
          >
            {pageError}
          </div>
        )}
        {isLoadingApplications && (
          <div className="mb-6 rounded-xl border border-blue-100 bg-blue-50 px-4 py-3 text-sm text-blue-700">
            Loading applications from CloudMorph...
          </div>
        )}

        {/* =================================================== */}
        {/* SUMMARY CARDS */}
        {/* =================================================== */}
        <div className="grid grid-cols-1 gap-5 md:grid-cols-2 xl:grid-cols-4">
          <div className="group rounded-2xl border border-slate-200 bg-white p-6 shadow-sm transition duration-300 hover:-translate-y-1 hover:shadow-xl">
            <div className="flex items-start justify-between">
              <div className="rounded-xl bg-blue-100 p-3 text-blue-600">
                <Cloud size={23} />
              </div>
              <span className="rounded-full bg-blue-50 px-3 py-1 text-xs font-semibold text-blue-600">
                Active
              </span>
            </div>
            <p className="mt-6 text-sm font-medium text-slate-500">
              Total Applications
            </p>
            <p className="mt-1 text-3xl font-bold text-slate-900">
              {applications.length}
            </p>
            <p className="mt-3 text-xs text-slate-400">
              Across AWS, Google Cloud and Azure
            </p>
          </div>

          <div className="group rounded-2xl border border-slate-200 bg-white p-6 shadow-sm transition duration-300 hover:-translate-y-1 hover:shadow-xl">
            <div className="flex items-start justify-between">
              <div className="rounded-xl bg-emerald-100 p-3 text-emerald-600">
                <ShieldCheck size={23} />
              </div>
              <span className="flex items-center gap-1 text-xs font-semibold text-emerald-600">
                <TrendingUp size={14} />
                Stable
              </span>
            </div>
            <p className="mt-6 text-sm font-medium text-slate-500">
              Healthy Applications
            </p>
            <p className="mt-1 text-3xl font-bold text-slate-900">
              {healthyApps}
            </p>
            <p className="mt-3 text-xs text-slate-400">
              {applications.length > 0
                ? Math.round((healthyApps / applications.length) * 100)
                : 0}
              % applications healthy
            </p>
          </div>

          <div className="group rounded-2xl border border-slate-200 bg-white p-6 shadow-sm transition duration-300 hover:-translate-y-1 hover:shadow-xl">
            <div className="flex items-start justify-between">
              <div className="rounded-xl bg-violet-100 p-3 text-violet-600">
                <CircleDollarSign size={23} />
              </div>
              <span className="flex items-center gap-1 text-xs font-semibold text-emerald-600">
                <TrendingUp size={14} />
                Tracked
              </span>
            </div>
            <p className="mt-6 text-sm font-medium text-slate-500">
              Total Monthly Cost
            </p>
            <p className="mt-1 text-3xl font-bold text-slate-900">
              ${totalCost.toFixed(2)}
            </p>
            <p className="mt-3 text-xs text-slate-400">
              Estimated across all applications
            </p>
          </div>

          <div className="group rounded-2xl border border-slate-200 bg-white p-6 shadow-sm transition duration-300 hover:-translate-y-1 hover:shadow-xl">
            <div className="flex items-start justify-between">
              <div className="rounded-xl bg-amber-100 p-3 text-amber-600">
                <Sparkles size={23} />
              </div>
              <span className="rounded-full bg-amber-50 px-3 py-1 text-xs font-semibold text-amber-600">
                AI Found
              </span>
            </div>
            <p className="mt-6 text-sm font-medium text-slate-500">
              Optimization Opportunities
            </p>
            <p className="mt-1 text-3xl font-bold text-slate-900">
              {applications.length}
            </p>
            <p className="mt-3 text-xs text-slate-400">
              Potential AI improvements detected
            </p>
          </div>
        </div>

        {/* =================================================== */}
        {/* AI INSIGHT */}
        {/* =================================================== */}
        <div className="mt-8 flex flex-col gap-5 rounded-2xl border border-blue-100 bg-gradient-to-r from-blue-50 via-white to-violet-50 p-6 lg:flex-row lg:items-center lg:justify-between">
          <div className="flex items-start gap-4">
            <div className="rounded-xl bg-gradient-to-br from-blue-600 to-violet-600 p-3 text-white shadow-lg shadow-blue-100">
              <BrainCircuit size={24} />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h2 className="text-lg font-bold text-slate-800">
                  CloudMorph AI Insight
                </h2>
                <span className="rounded-full bg-blue-100 px-2 py-0.5 text-[10px] font-bold text-blue-600">
                  INSIGHTS
                </span>
              </div>
              <p className="mt-2 max-w-2xl text-sm leading-relaxed text-slate-500">
                CloudMorph detected{" "}
                <span className="font-semibold text-slate-700">
                  {applications.length} optimization opportunities
                </span>{" "}
                that could potentially reduce your cloud spending by{" "}
                <span className="font-semibold text-emerald-600">
                  approximately ${totalSavings}/month.
                </span>
              </p>
            </div>
          </div>
          <button
            type="button"
            onClick={() => navigate("/app/optimizer")}
            className="flex shrink-0 items-center justify-center gap-2 rounded-xl bg-blue-600 px-5 py-3 text-sm font-semibold text-white shadow-lg shadow-blue-200 transition hover:bg-blue-700"
          >
            View AI Analysis
            <ArrowUpRight size={17} />
          </button>
        </div>

        {/* =================================================== */}
        {/* FILTERS */}
        {/* =================================================== */}
        <div className="mt-8 rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
          <div className="flex flex-col gap-4 xl:flex-row xl:items-center xl:justify-between">
            <div className="flex w-full max-w-xl items-center gap-3 rounded-xl border border-slate-200 bg-slate-50 px-4 py-3 transition focus-within:border-blue-400 focus-within:bg-white focus-within:ring-4 focus-within:ring-blue-50">
              <Search size={19} className="text-slate-400" />
              <input
                type="text"
                value={searchTerm}
                onChange={(event) => setSearchTerm(event.target.value)}
                placeholder="Search applications, providers, or categories..."
                className="w-full bg-transparent text-sm text-slate-700 outline-none placeholder:text-slate-400"
              />
              {searchTerm && (
                <button
                  type="button"
                  onClick={() => setSearchTerm("")}
                  className="text-slate-400 hover:text-slate-700"
                >
                  <X size={17} />
                </button>
              )}
            </div>
            <div className="flex flex-wrap items-center gap-3">
              <div className="flex items-center gap-2 text-slate-500">
                <SlidersHorizontal size={18} />
                <span className="text-sm font-medium">Filters</span>
              </div>
              <div className="relative">
                <select
                  value={providerFilter}
                  onChange={(event) => setProviderFilter(event.target.value)}
                  className="appearance-none rounded-xl border border-slate-200 bg-white px-4 py-2.5 pr-9 text-sm text-slate-600 outline-none transition hover:border-slate-300 focus:border-blue-500"
                >
                  <option value="All">All Providers</option>
                  <option value="AWS">AWS</option>
                  <option value="GCP">GCP</option>
                  <option value="Azure">Azure</option>
                </select>
                <ChevronDown
                  size={16}
                  className="pointer-events-none absolute right-3 top-3 text-slate-400"
                />
              </div>
              <div className="relative">
                <select
                  value={statusFilter}
                  onChange={(event) => setStatusFilter(event.target.value)}
                  className="appearance-none rounded-xl border border-slate-200 bg-white px-4 py-2.5 pr-9 text-sm text-slate-600 outline-none transition hover:border-slate-300 focus:border-blue-500"
                >
                  <option value="All">All Status</option>
                  <option value="Healthy">Healthy</option>
                  <option value="Warning">Warning</option>
                </select>
                <ChevronDown
                  size={16}
                  className="pointer-events-none absolute right-3 top-3 text-slate-400"
                />
              </div>
              <div className="relative">
                <select
                  value={sortBy}
                  onChange={(event) => setSortBy(event.target.value)}
                  className="appearance-none rounded-xl border border-slate-200 bg-white px-4 py-2.5 pr-9 text-sm text-slate-600 outline-none transition hover:border-slate-300 focus:border-blue-500"
                >
                  <option>Cost High to Low</option>
                  <option>Cost Low to High</option>
                  <option>Name A-Z</option>
                  <option>Health Score</option>
                </select>
                <ChevronDown
                  size={16}
                  className="pointer-events-none absolute right-3 top-3 text-slate-400"
                />
              </div>
              {(searchTerm ||
                providerFilter !== "All" ||
                statusFilter !== "All") && (
                <button
                  type="button"
                  onClick={clearFilters}
                  className="flex items-center gap-2 rounded-xl px-3 py-2.5 text-sm font-medium text-red-500 transition hover:bg-red-50"
                >
                  <X size={16} />
                  Clear
                </button>
              )}
            </div>
          </div>
          <div className="mt-5 flex items-center justify-between border-t border-slate-100 pt-4">
            <p className="text-sm text-slate-500">
              Showing{" "}
              <span className="font-semibold text-slate-700">
                {filteredApplications.length}
              </span>{" "}
              of{" "}
              <span className="font-semibold text-slate-700">
                {applications.length}
              </span>{" "}
              applications
            </p>
            {warningApps > 0 && (
              <div className="flex items-center gap-2 text-xs font-medium text-amber-600">
                <AlertTriangle size={15} />
                {warningApps} application requires attention
              </div>
            )}
          </div>
        </div>

        {/* =================================================== */}
        {/* APPLICATION HEADER */}
        {/* =================================================== */}
        <div className="mt-8 flex items-center justify-between">
          <div>
            <h2 className="text-2xl font-bold text-slate-900">
              Your Applications
            </h2>
            <p className="mt-1 text-sm text-slate-500">
              Monitor performance, costs, and AI optimization opportunities.
            </p>
          </div>
          <button
            type="button"
            onClick={() => navigate("/app/analytics")}
            className="hidden items-center gap-2 text-sm font-semibold text-blue-600 md:flex"
          >
            <BarChart3 size={17} />
            View Cost Analytics
          </button>
        </div>

        {/* =================================================== */}
        {/* APPLICATION GRID */}
        {/* =================================================== */}
        {isLoadingApplications ? (
          <div className="mt-6 rounded-2xl border border-slate-200 bg-white p-16 text-center text-sm text-slate-500">
            Loading your applications...
          </div>
        ) : filteredApplications.length > 0 ? (
          <div className="mt-6 grid grid-cols-1 gap-6 xl:grid-cols-2">
            {filteredApplications.map((application) => (
              <div
                key={application.id}
                className="group relative overflow-hidden rounded-2xl border border-slate-200 bg-white p-6 shadow-sm transition duration-300 hover:-translate-y-1 hover:shadow-xl"
              >
                <div
                  className={`absolute right-0 top-0 h-32 w-32 rounded-bl-full opacity-60 ${
                    application.status === "Warning"
                      ? "bg-amber-50"
                      : "bg-blue-50"
                  }`}
                />
                <div className="relative flex items-start justify-between">
                  <div className="flex items-start gap-4">
                    <div className="flex h-14 w-14 shrink-0 items-center justify-center rounded-2xl bg-gradient-to-br from-blue-600 to-violet-600 text-white shadow-lg shadow-blue-200">
                      <Cloud size={26} />
                    </div>
                    <div>
                      <div className="flex flex-wrap items-center gap-2">
                        <h2 className="text-lg font-bold text-slate-800">
                          {application.name}
                        </h2>
                        <span className="rounded-full bg-slate-100 px-2.5 py-1 text-[10px] font-bold uppercase tracking-wide text-slate-500">
                          {application.category}
                        </span>
                      </div>
                      <p className="mt-1 text-sm text-slate-500">
                        {application.description}
                      </p>
                    </div>
                  </div>
                  <div className="relative z-10">
                    <button
                      type="button"
                      aria-label={`More options for ${application.name}`}
                      aria-expanded={activeMenuId === application.id}
                      onClick={() =>
                        setActiveMenuId((current) =>
                          current === application.id ? null : application.id
                        )
                      }
                      className="rounded-lg p-2 text-slate-400 transition hover:bg-slate-100 hover:text-slate-700"
                    >
                      <MoreVertical size={20} />
                    </button>
                    {activeMenuId === application.id && (
                      <div className="absolute right-0 top-11 z-20 w-44 overflow-hidden rounded-xl border border-slate-200 bg-white py-1 shadow-xl">
                        <button
                          type="button"
                          onClick={() => {
                            setActiveMenuId(null);
                            navigate(`/app/applications/${application.id}`);
                          }}
                          className="block w-full px-4 py-2.5 text-left text-sm text-slate-700 transition hover:bg-slate-50"
                        >
                          View details
                        </button>
                        <button
                          type="button"
                          disabled={deletingApplicationId === application.id}
                          onClick={() => handleDeleteApplication(application)}
                          className="block w-full px-4 py-2.5 text-left text-sm text-red-600 transition hover:bg-red-50 disabled:opacity-60"
                        >
                          {deletingApplicationId === application.id
                            ? "Deleting..."
                            : "Delete application"}
                        </button>
                      </div>
                    )}
                  </div>
                </div>
                <div className="relative mt-6 flex flex-wrap gap-2.5">
                  <span
                    className={`rounded-full px-3 py-1.5 text-xs font-bold ${
                      application.provider === "AWS"
                        ? "bg-orange-50 text-orange-600"
                        : application.provider === "Azure"
                          ? "bg-cyan-50 text-cyan-600"
                          : "bg-blue-50 text-blue-600"
                    }`}
                  >
                    {application.provider}
                  </span>
                  <span
                    className={`flex items-center gap-1.5 rounded-full px-3 py-1.5 text-xs font-semibold ${
                      application.status === "Healthy"
                        ? "bg-emerald-50 text-emerald-600"
                        : "bg-amber-50 text-amber-600"
                    }`}
                  >
                    {application.status === "Healthy" ? (
                      <CheckCircle2 size={14} />
                    ) : (
                      <AlertTriangle size={14} />
                    )}
                    {application.status}
                  </span>
                  <span
                    className={`rounded-full px-3 py-1.5 text-xs font-semibold ${
                      application.usage === "High"
                        ? "bg-red-50 text-red-600"
                        : application.usage === "Medium"
                          ? "bg-blue-50 text-blue-600"
                          : "bg-slate-100 text-slate-600"
                    }`}
                  >
                    {application.usage} Usage
                  </span>
                </div>
                <div className="relative mt-6 grid grid-cols-3 gap-3">
                  <div className="rounded-xl bg-slate-50 p-4">
                    <div className="flex items-center gap-1.5 text-slate-400">
                      <Server size={15} />
                      <span className="text-xs">Services</span>
                    </div>
                    <p className="mt-2 text-lg font-bold text-slate-800">
                      {application.services}
                    </p>
                  </div>
                  <div className="rounded-xl bg-slate-50 p-4">
                    <div className="flex items-center gap-1.5 text-slate-400">
                      <DollarSign size={15} />
                      <span className="text-xs">Cost</span>
                    </div>
                    <p className="mt-2 text-lg font-bold text-slate-800">
                      ${application.monthlyCost.toFixed(2)}
                    </p>
                  </div>
                  <div className="rounded-xl bg-slate-50 p-4">
                    <div className="flex items-center gap-1.5 text-slate-400">
                      <ShieldCheck size={15} />
                      <span className="text-xs">Health</span>
                    </div>
                    <p
                      className={`mt-2 text-lg font-bold ${
                        application.healthScore >= 85
                          ? "text-emerald-600"
                          : "text-amber-600"
                      }`}
                    >
                      {application.healthScore}%
                    </p>
                  </div>
                </div>
                <div className="relative mt-6">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <Activity size={16} className="text-slate-400" />
                      <span className="text-sm font-medium text-slate-600">
                        Resource Utilization
                      </span>
                    </div>
                    <span className="text-sm font-bold text-slate-700">
                      {application.usagePercent === null
                        ? "Not Available"
                        : `${application.usagePercent}%`}
                    </span>
                  </div>
                  <div className="mt-3 h-2 overflow-hidden rounded-full bg-slate-100">
                    <div
                      className={`h-full rounded-full ${
                        application.usagePercent === null
                          ? "bg-slate-300"
                          : application.usagePercent >= 80
                            ? "bg-red-500"
                            : application.usagePercent >= 50
                              ? "bg-blue-500"
                              : "bg-emerald-500"
                      }`}
                      style={{ width: `${application.usagePercent ?? 0}%` }}
                    />
                  </div>
                </div>
                <div className="relative mt-6 rounded-xl border border-violet-100 bg-gradient-to-r from-violet-50 to-blue-50 p-4">
                  <div className="flex items-start justify-between gap-4">
                    <div className="flex gap-3">
                      <div className="rounded-lg bg-violet-100 p-2 text-violet-600">
                        <Sparkles size={18} />
                      </div>
                      <div>
                        <p className="text-xs font-bold uppercase tracking-wide text-violet-600">
                          AI Recommendation
                        </p>
                        <p className="mt-1 text-sm font-medium text-slate-700">
                          {application.recommendation}
                        </p>
                        {application.hasLatestOptimization && (
                          <div className="mt-2 flex flex-wrap items-center gap-2 text-xs">
                            {application.latestOptimizationAction && (
                              <span className="rounded-full bg-blue-100 px-2.5 py-1 font-bold text-blue-700">
                                Decision: {application.latestOptimizationAction}
                              </span>
                            )}
                            {application.workloadPrediction && (
                              <span className="rounded-full bg-violet-100 px-2.5 py-1 font-semibold text-violet-700">
                                Workload: {application.workloadPrediction}
                              </span>
                            )}
                            {application.workloadConfidence !== null && (
                              <span className="rounded-full bg-slate-100 px-2.5 py-1 font-semibold text-slate-600">
                                ML confidence: {application.workloadConfidence.toFixed(2)}%
                              </span>
                            )}
                          </div>
                        )}
                        {application.decisionSource && (
                          <p className="mt-2 text-[11px] text-slate-500">
                            Decision source: {application.decisionSource}
                          </p>
                        )}
                      </div>
                    </div>
                    <div className="text-right">
                      <p className="text-xs text-slate-500">Save up to</p>
                      <p className="mt-1 text-sm font-bold text-emerald-600">
                        ${application.potentialSavings}/mo
                      </p>
                    </div>
                  </div>
                </div>
                <div className="relative mt-5 flex items-center justify-between border-t border-slate-100 pt-5">
                  <div className="flex items-center gap-2">
                    <span
                      className={`h-2 w-2 rounded-full ${
                        application.status === "Healthy"
                          ? "bg-emerald-500"
                          : "bg-amber-500"
                      }`}
                    />
                    <span className="text-xs text-slate-400">
                      Saved application data
                    </span>
                  </div>
                  <button
                    type="button"
                    onClick={() =>
                      navigate(`/app/applications/${application.id}`)
                    }
                    className="flex items-center gap-2 rounded-xl bg-slate-900 px-4 py-2.5 text-sm font-semibold text-white transition hover:bg-blue-600 group-hover:shadow-lg"
                  >
                    View Details
                    <ChevronRight size={17} />
                  </button>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="mt-6 rounded-2xl border border-dashed border-slate-300 bg-white p-16 text-center">
            <div className="mx-auto flex h-16 w-16 items-center justify-center rounded-2xl bg-slate-100 text-slate-400">
              <Search size={28} />
            </div>
            <h3 className="mt-5 text-lg font-bold text-slate-800">
              No applications found
            </h3>
            <p className="mt-2 text-sm text-slate-500">
              Try changing your search or filter settings.
            </p>
            <button
              type="button"
              onClick={clearFilters}
              className="mt-5 rounded-xl bg-blue-600 px-5 py-3 text-sm font-semibold text-white transition hover:bg-blue-700"
            >
              Clear Filters
            </button>
          </div>
        )}

        {/* =================================================== */}
        {/* BOTTOM SUMMARY */}
        {/* =================================================== */}
        <div className="mt-8 grid grid-cols-1 gap-5 lg:grid-cols-3">
          <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
            <div className="flex items-center gap-3">
              <div className="rounded-xl bg-blue-100 p-2.5 text-blue-600">
                <Server size={20} />
              </div>
              <div>
                <p className="text-sm font-semibold text-slate-700">
                  {totalServices} Cloud Services
                </p>
                <p className="mt-1 text-xs text-slate-400">
                  Currently monitored across applications
                </p>
              </div>
            </div>
          </div>
          <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
            <div className="flex items-center gap-3">
              <div className="rounded-xl bg-emerald-100 p-2.5 text-emerald-600">
                <DollarSign size={20} />
              </div>
              <div>
                <p className="text-sm font-semibold text-slate-700">
                  ${totalSavings} Potential Savings
                </p>
                <p className="mt-1 text-xs text-slate-400">
                  Identified through AI optimization analysis
                </p>
              </div>
            </div>
          </div>
          <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
            <div className="flex items-center gap-3">
              <div className="rounded-xl bg-violet-100 p-2.5 text-violet-600">
                <Cpu size={20} />
              </div>
              <div>
                <p className="text-sm font-semibold text-slate-700">
                  AI Analysis Ready
                </p>
                <p className="mt-1 text-xs text-slate-400">
                  CloudMorph can analyze your infrastructure configuration
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* ===================================================== */}
      {/* ADD APPLICATION MODAL */}
      {/* ===================================================== */}
      <AddApplicationModal
        isOpen={isModalOpen}
        onClose={() => setIsModalOpen(false)}
        onAdd={handleAddApplication}
      />
    </div>
  );
}

/* ========================================================= */
/* ADD APPLICATION MODAL */
/* ========================================================= */
function AddApplicationModal({ isOpen, onClose, onAdd }) {
  const [formData, setFormData] = useState({
    name: "",
    description: "",
    provider: "AWS",
    category: "Web Application",
    services: "",
    monthlyCost: "",
    utilization: "",
    status: "Healthy",
  });
  const [isSubmitting, setIsSubmitting] = useState(false);

  if (!isOpen) {
    return null;
  }

  const handleChange = (event) => {
    const { name, value } = event.target;
    setFormData((previousData) => ({
      ...previousData,
      [name]: value,
    }));
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    if (isSubmitting) return;
    if (
      !formData.name.trim() ||
      !formData.description.trim() ||
      !formData.services ||
      !formData.monthlyCost ||
      !Number.isFinite(Number(formData.services)) ||
      !Number.isInteger(Number(formData.services)) ||
      Number(formData.services) < 1 ||
      !Number.isFinite(Number(formData.monthlyCost)) ||
      Number(formData.monthlyCost) < 0 ||
      (formData.utilization.trim() !== "" &&
        (!Number.isInteger(Number(formData.utilization)) ||
          Number(formData.utilization) < 0 ||
          Number(formData.utilization) > 100))
    ) {
      return;
    }
    setIsSubmitting(true);
    try {
      const saved = await onAdd(formData);
      if (!saved) return;
      setFormData({
        name: "",
        description: "",
        provider: "AWS",
        category: "Web Application",
        services: "",
        monthlyCost: "",
        utilization: "",
        status: "Healthy",
      });
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/70 p-4 backdrop-blur-sm">
      <div className="relative w-full max-w-2xl overflow-hidden rounded-3xl bg-white shadow-2xl">
        {/* MODAL HEADER */}
        <div className="relative overflow-hidden bg-gradient-to-r from-slate-950 via-blue-950 to-violet-950 px-7 py-6">
          <div className="absolute -right-10 -top-10 h-32 w-32 rounded-full bg-blue-500/20 blur-2xl" />
          <div className="relative flex items-start justify-between">
            <div>
              <div className="flex items-center gap-2">
                <div className="rounded-lg bg-white/10 p-2 text-blue-200">
                  <Cloud size={20} />
                </div>
                <span className="text-xs font-bold uppercase tracking-widest text-blue-300">
                  New Cloud Application
                </span>
              </div>
              <h2 className="mt-4 text-2xl font-bold text-white">
                Add Application
              </h2>
              <p className="mt-2 text-sm text-slate-300">
                Add your application to CloudMorph for monitoring and AI-powered
                cloud optimization.
              </p>
            </div>
            <button
              type="button"
              onClick={onClose}
              className="rounded-xl bg-white/10 p-2 text-slate-300 transition hover:bg-white/20 hover:text-white"
            >
              <X size={20} />
            </button>
          </div>
        </div>

        {/* FORM */}
        <form
          onSubmit={handleSubmit}
          className="max-h-[70vh] overflow-y-auto p-7"
        >
          <div className="grid grid-cols-1 gap-5 md:grid-cols-2">
            {/* NAME */}
            <div className="md:col-span-2">
              <label className="text-sm font-semibold text-slate-700">
                Application Name
              </label>
              <input
                type="text"
                name="name"
                required
                value={formData.name}
                onChange={handleChange}
                placeholder="Example: E-Commerce Platform"
                className="mt-2 w-full rounded-xl border border-slate-200 bg-slate-50 px-4 py-3 text-sm outline-none transition focus:border-blue-500 focus:bg-white focus:ring-4 focus:ring-blue-50"
              />
            </div>

            {/* DESCRIPTION */}
            <div className="md:col-span-2">
              <label className="text-sm font-semibold text-slate-700">
                Description
              </label>
              <textarea
                name="description"
                required
                value={formData.description}
                onChange={handleChange}
                rows="3"
                placeholder="Briefly describe your application..."
                className="mt-2 w-full resize-none rounded-xl border border-slate-200 bg-slate-50 px-4 py-3 text-sm outline-none transition focus:border-blue-500 focus:bg-white focus:ring-4 focus:ring-blue-50"
              />
            </div>

            {/* PROVIDER */}
            <div>
              <label className="text-sm font-semibold text-slate-700">
                Cloud Provider
              </label>
              <select
                name="provider"
                value={formData.provider}
                onChange={handleChange}
                className="mt-2 w-full rounded-xl border border-slate-200 bg-slate-50 px-4 py-3 text-sm outline-none transition focus:border-blue-500 focus:bg-white"
              >
                <option value="AWS">AWS</option>
                <option value="GCP">Google Cloud</option>
                <option value="Azure">Microsoft Azure</option>
              </select>
            </div>

            {/* CATEGORY */}
            <div>
              <label className="text-sm font-semibold text-slate-700">
                Application Category
              </label>
              <select
                name="category"
                value={formData.category}
                onChange={handleChange}
                className="mt-2 w-full rounded-xl border border-slate-200 bg-slate-50 px-4 py-3 text-sm outline-none transition focus:border-blue-500 focus:bg-white"
              >
                <option>Web Application</option>
                <option>Backend API</option>
                <option>Mobile Application</option>
                <option>Microservice</option>
                <option>Data Analytics</option>
                <option>Machine Learning</option>
              </select>
            </div>

            {/* SERVICES */}
            <div>
              <label className="text-sm font-semibold text-slate-700">
                Number of Cloud Services
              </label>
              <input
                type="number"
                name="services"
                required
                value={formData.services}
                onChange={handleChange}
                min="1"
                step="1"
                placeholder="Example: 5"
                className="mt-2 w-full rounded-xl border border-slate-200 bg-slate-50 px-4 py-3 text-sm outline-none transition focus:border-blue-500 focus:bg-white focus:ring-4 focus:ring-blue-50"
              />
            </div>

            {/* COST */}
            <div>
              <label className="text-sm font-semibold text-slate-700">
                Estimated Monthly Cost ($)
              </label>
              <input
                type="number"
                name="monthlyCost"
                required
                value={formData.monthlyCost}
                onChange={handleChange}
                min="0"
                step="0.01"
                placeholder="Example: 120"
                className="mt-2 w-full rounded-xl border border-slate-200 bg-slate-50 px-4 py-3 text-sm outline-none transition focus:border-blue-500 focus:bg-white focus:ring-4 focus:ring-blue-50"
              />
            </div>

            {/* RESOURCE UTILIZATION */}
            <div className="md:col-span-2">
              <label className="text-sm font-semibold text-slate-700">
                Resource Utilization (%)
                <span className="ml-2 text-xs font-normal text-slate-400">
                  Optional
                </span>
              </label>
              <input
                type="number"
                name="utilization"
                value={formData.utilization}
                onChange={handleChange}
                min="0"
                max="100"
                step="1"
                placeholder="Example: 65"
                className="mt-2 w-full rounded-xl border border-slate-200 bg-slate-50 px-4 py-3 text-sm outline-none transition focus:border-blue-500 focus:bg-white focus:ring-4 focus:ring-blue-50"
              />
              <p className="mt-2 text-xs text-slate-400">
                Enter a value between 0 and 100. Leave empty if utilization data is unavailable.
              </p>
            </div>

            {/* STATUS */}
            <div className="md:col-span-2">
              <label className="text-sm font-semibold text-slate-700">
                Initial Application Status
              </label>
              <div className="mt-3 grid grid-cols-2 gap-3">
                <button
                  type="button"
                  onClick={() =>
                    setFormData((previousData) => ({
                      ...previousData,
                      status: "Healthy",
                    }))
                  }
                  className={`rounded-xl border p-4 text-left transition ${
                    formData.status === "Healthy"
                      ? "border-emerald-400 bg-emerald-50"
                      : "border-slate-200 hover:border-slate-300"
                  }`}
                >
                  <div className="flex items-center gap-2">
                    <CheckCircle2 size={18} className="text-emerald-600" />
                    <span className="text-sm font-semibold text-slate-700">
                      Healthy
                    </span>
                  </div>
                  <p className="mt-2 text-xs text-slate-500">
                    Application is running normally.
                  </p>
                </button>
                <button
                  type="button"
                  onClick={() =>
                    setFormData((previousData) => ({
                      ...previousData,
                      status: "Warning",
                    }))
                  }
                  className={`rounded-xl border p-4 text-left transition ${
                    formData.status === "Warning"
                      ? "border-amber-400 bg-amber-50"
                      : "border-slate-200 hover:border-slate-300"
                  }`}
                >
                  <div className="flex items-center gap-2">
                    <AlertTriangle size={18} className="text-amber-600" />
                    <span className="text-sm font-semibold text-slate-700">
                      Warning
                    </span>
                  </div>
                  <p className="mt-2 text-xs text-slate-500">
                    Application requires attention.
                  </p>
                </button>
              </div>
            </div>
          </div>

          {/* BUTTONS */}
          <div className="mt-7 flex flex-col-reverse gap-3 border-t border-slate-100 pt-5 sm:flex-row sm:justify-end">
            <button
              type="button"
              onClick={onClose}
              className="rounded-xl px-5 py-3 text-sm font-semibold text-slate-600 transition hover:bg-slate-100"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={isSubmitting}
              className="flex items-center justify-center gap-2 rounded-xl bg-gradient-to-r from-blue-600 to-violet-600 px-6 py-3 text-sm font-semibold text-white shadow-lg shadow-blue-200 transition hover:shadow-xl disabled:cursor-not-allowed disabled:opacity-70"
            >
              {isSubmitting ? (
                <>
                  <RefreshCw size={18} className="animate-spin" />
                  Saving...
                </>
              ) : (
                <>
                  <Plus size={18} />
                  Add Application
                </>
              )}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}

export default Applications;