import { useCallback, useEffect, useMemo, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import API from "../services/api";
import {
  getOptimizationHistory,
  getOptimizationError,
} from "../services/optimization";

import {
  BrainCircuit,
  Sparkles,
  Search,
  Filter,
  ArrowUpRight,
  ArrowRight,
  TrendingDown,
  TrendingUp,
  DollarSign,
  Target,
  Activity,
  Cpu,
  ShieldCheck,
  RefreshCw,
  Zap,
  CheckCircle2,
  Cloud,
  ChevronDown,
  SlidersHorizontal,
  BarChart3,
  CircleDollarSign,
  Lightbulb,
  Server,
  Layers3,
  Database,
  HardDrive,
  Network,
  Clock3,
  AlertTriangle,
  X,
  Gauge,
  GitBranch,
  ShieldAlert,
  Workflow,
  CircleCheck,
  Trash2,
} from "lucide-react";

const toFiniteNumber = (value, fallback = null) => {
  if (value === null || value === undefined || value === "") return fallback;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
};

const clampPercent = (value) => {
  const parsed = toFiniteNumber(value, null);
  if (parsed === null) return null;
  const normalized = parsed > 0 && parsed <= 1 ? parsed * 100 : parsed;
  return Math.max(0, Math.min(100, normalized));
};

const formatScore = (value) =>
  Number.isFinite(value) ? `${Math.round(value)}%` : "Pending";

function AIAnalysis() {
  const navigate = useNavigate();
  const { id: routeApplicationId } = useParams();

  /* ========================================================= */
  /* STATE */
  /* ========================================================= */

  const [searchQuery, setSearchQuery] = useState("");
  const [providerFilter, setProviderFilter] = useState("All");
  const [sortBy, setSortBy] = useState("score");

  const [isRefreshing, setIsRefreshing] = useState(false);
  const [isRunningDecision, setIsRunningDecision] =
    useState(false);

  const [isRunningOptimization, setIsRunningOptimization] =
    useState(false);
  const [optimizationResult, setOptimizationResult] = useState(null);
  const [optimizationHistory, setOptimizationHistory] = useState([]);
  const [optimizationError, setOptimizationError] = useState("");
  const [selectedMachineId, setSelectedMachineId] = useState("1");

  const [lastUpdated, setLastUpdated] = useState(
    new Date().toLocaleTimeString([], {
      hour: "2-digit",
      minute: "2-digit",
    })
  );

  const [selectedApplication, setSelectedApplication] =
    useState(null);
  const [quickViewApplication, setQuickViewApplication] =
    useState(null);

  const [selectedDecision, setSelectedDecision] =
    useState("RESIZE");

  /* ========================================================= */
  /* APPLICATION DATA */
  /* ========================================================= */

  const [applications, setApplications] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [loadError, setLoadError] = useState("");

  const loadApplications = useCallback(async ({ refresh = false } = {}) => {
    if (refresh) setIsRefreshing(true);
    else setIsLoading(true);
    setLoadError("");
    try {
      const response = await API.get("/applications");
      const payload = response.data;
      const records = Array.isArray(payload)
        ? payload
        : payload?.applications ?? payload?.data ?? [];
      if (!Array.isArray(records)) throw new Error("Unexpected applications response");

      const mapped = records
        .filter((app) => app && typeof app === "object")
        .map((app) => {
          const monthlyCost = Math.max(
            0,
            toFiniteNumber(app.cost ?? app.monthly_cost ?? app.monthlyCost, 0)
          );
          const optimizedRaw = toFiniteNumber(
            app.optimized_cost ?? app.optimizedCost ?? app.optimized_cost_monthly,
            null
          );
          const optimizedCost =
            optimizedRaw === null ? monthlyCost : Math.max(0, optimizedRaw);
          const usagePercent = clampPercent(
            app.utilization ?? app.usagePercent ?? app.utilization_percent
          );
          const savedSavings = toFiniteNumber(
            app.potential_savings ?? app.potentialSavings,
            null
          );
          const potentialSavings = Math.max(
            0,
            savedSavings === null
              ? monthlyCost - optimizedCost
              : savedSavings
          );
          // Prefer the latest persisted ML recommendation when available.
          // Keep the existing application decision as a compatibility fallback.
          const latestOptimization =
            app.latest_optimization ?? app.latestOptimization ?? null;
          const rawDecision = String(
            latestOptimization?.action ?? app.decision ?? ""
          ).toUpperCase();
          const decision = ["KEEP", "RESIZE", "SWITCH"].includes(rawDecision)
            ? rawDecision
            : "KEEP";
          const rawStatus = String(app.status || "Active").toLowerCase();
          const category = String(app.category || app.service || "Cloud Application");
          const healthScore = clampPercent(app.health_score ?? app.healthScore);
          const healthScoreStatus = String(
            app.health_score_status ?? app.healthScoreStatus ?? "pending"
          ).toLowerCase();
          const optimizationScore = clampPercent(
            app.optimization_score ?? app.optimizationScore
          );
          const optimizationScoreType = String(
            app.optimization_score_type ??
            app.optimizationScoreType ??
            "utilization_fit_rule_based"
          );
          const providerRaw = String(app.provider || "Unknown");
          const providerLower = providerRaw.toLowerCase();
          const provider = providerLower.includes("google") || providerLower === "gcp"
            ? "GCP"
            : providerLower.includes("azure")
              ? "Azure"
              : providerLower.includes("aws")
                ? "AWS"
                : providerRaw;
          const resources = {
            compute: clampPercent(app.compute_utilization ?? app.cpu_utilization ?? app.resources?.compute),
            database: clampPercent(app.database_utilization ?? app.resources?.database),
            storage: clampPercent(app.storage_utilization ?? app.resources?.storage),
            network: clampPercent(app.network_utilization ?? app.resources?.network),
          };
          const id = String(app.id ?? app._id ?? "");
          return {
            id,
            name: String(app.name || "Unnamed Application"),
            description: String(app.description || "Cloud application"),
            provider,
            services: Math.max(0, toFiniteNumber(app.services ?? app.service_count, 0)),
            monthlyCost,
            optimizedCost,
            status: ["healthy", "active", "running", "ok"].includes(rawStatus)
              ? "Healthy"
              : ["warning", "unhealthy", "critical", "error"].includes(rawStatus)
                ? "Warning"
                : "Unknown",
            usage: usagePercent === null
              ? "Not available"
              : usagePercent >= 85
                ? "High"
                : usagePercent >= 60
                  ? "Medium"
                  : "Low",
            usagePercent,
            healthScore,
            healthScoreStatus,
            optimizationScore,
            optimizationScoreType,
            recommendation: String(
              latestOptimization?.title ||
              app.recommendation ||
              app.ai_recommendation ||
              `Saved recommendation: ${decision}`
            ),
            latestOptimization,
            decision,
            potentialSavings,
            category,
            icon: Server,
            trend: "up",
            resources,
          };
        })
        .filter((app) => app.id);
      setApplications(mapped);
      setLastUpdated(new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }));
    } catch (error) {
      const message =
        error.response?.status === 401
          ? "Your session has expired. Please log in again."
          : error.response?.data?.detail ||
            (error.response
              ? "Unable to load applications. Please try again."
              : "Unable to connect to CloudMorph API. Check that the backend is running.");
      setLoadError(String(message));
    } finally {
      setIsLoading(false);
      setIsRefreshing(false);
    }
  }, []);

  useEffect(() => { loadApplications(); }, [loadApplications]);


  /* ========================================================= */
  /* CALCULATIONS */
  /* ========================================================= */

  const totalMonthlyCost = applications.reduce(
    (total, application) =>
      total + application.monthlyCost,
    0
  );

  const totalPotentialSavings = applications.reduce(
    (total, application) =>
      total + application.potentialSavings,
    0
  );

  const knownOptimizationScores = applications
    .map((application) => application.optimizationScore)
    .filter(Number.isFinite);
  const knownHealthScores = applications
    .map((application) => application.healthScore)
    .filter(Number.isFinite);

  const averageOptimizationScore = knownOptimizationScores.length
    ? Math.round(
        knownOptimizationScores.reduce((total, score) => total + score, 0) /
          knownOptimizationScores.length
      )
    : null;

  const averageHealthScore = knownHealthScores.length
    ? Math.round(
        knownHealthScores.reduce((total, score) => total + score, 0) /
          knownHealthScores.length
      )
    : null;

  const totalServices = applications.reduce(
    (total, application) =>
      total + application.services,
    0
  );

  const highPriorityApps = applications.filter(
    (application) =>
      Number.isFinite(application.optimizationScore) &&
      application.optimizationScore < 75
  ).length;

  const healthyApplications = applications.filter(
    (application) =>
      application.status === "Healthy"
  ).length;

  /* ========================================================= */
  /* FILTER + SORT */
  /* ========================================================= */

  const filteredApplications = useMemo(() => {
    const filtered = applications.filter(
      (application) => {
        const query = searchQuery.toLowerCase().trim();

        const matchesSearch =
          application.name
            .toLowerCase()
            .includes(query) ||
          application.description
            .toLowerCase()
            .includes(query) ||
          application.category
            .toLowerCase()
            .includes(query) ||
          application.provider
            .toLowerCase()
            .includes(query);

        const matchesProvider =
          providerFilter === "All" ||
          application.provider === providerFilter;

        return (
          matchesSearch &&
          matchesProvider
        );
      }
    );

    filtered.sort((a, b) => {
      if (sortBy === "score") {
        const aHasScore = Number.isFinite(a.optimizationScore);
        const bHasScore = Number.isFinite(b.optimizationScore);
        if (!aHasScore && !bHasScore) return 0;
        if (!aHasScore) return 1;
        if (!bHasScore) return -1;
        return a.optimizationScore - b.optimizationScore;
      }

      if (sortBy === "savings") {
        return (
          b.potentialSavings -
          a.potentialSavings
        );
      }

      if (sortBy === "cost") {
        return (
          b.monthlyCost -
          a.monthlyCost
        );
      }

      if (sortBy === "health") {
        const aHasScore = Number.isFinite(a.healthScore);
        const bHasScore = Number.isFinite(b.healthScore);
        if (!aHasScore && !bHasScore) return 0;
        if (!aHasScore) return 1;
        if (!bHasScore) return -1;
        return a.healthScore - b.healthScore;
      }

      return 0;
    });

    return filtered;
  }, [
    searchQuery,
    providerFilter,
    sortBy,
    applications,
  ]);

  /* ========================================================= */
  /* TOP OPPORTUNITY */
  /* ========================================================= */

  // Keep the opportunity panel aligned with the application the user selected.
  // When nothing has been selected yet, show the application with the highest savings.
  const topOpportunity =
    applications.find((app) => app.id === selectedApplication?.id) ||
    [...applications].sort(
      (a, b) => b.potentialSavings - a.potentialSavings
    )[0];

  /* ========================================================= */
  /* ACTIVE APPLICATION */
  /* ========================================================= */

  // When this page is opened from an application-specific route,
  // select that application instead of leaving the optimizer on the top opportunity.
  useEffect(() => {
    if (!routeApplicationId || applications.length === 0) return;

    const routeApplication = applications.find(
      (app) => app.id === routeApplicationId
    );

    if (routeApplication) {
      setSelectedApplication(routeApplication);
      setSelectedDecision(routeApplication.decision || "KEEP");
    }
  }, [routeApplicationId, applications]);

  const activeApplication =
    applications.find((app) => app.id === selectedApplication?.id) || topOpportunity;

  useEffect(() => {
    if (activeApplication) setSelectedDecision(activeApplication.decision || "KEEP");
  }, [activeApplication?.id, activeApplication?.decision]);

  /* ========================================================= */
  /* NAVIGATION */
  /* ========================================================= */

  const handleAnalyze = (application) => {
    const applicationId = String(application?.id ?? "").trim();

    if (!applicationId) {
      setLoadError(
        "Unable to analyze this application because its ID is missing. Refresh the applications list and try again."
      );
      return;
    }

    setLoadError("");
    setSelectedApplication(application);
    setSelectedDecision(application.decision || "KEEP");
    setQuickViewApplication(null);

    navigate(
      `/app/applications/${encodeURIComponent(applicationId)}/analysis`,
      { state: { application } }
    );
  };

  const handleViewDetails = (application) => {
    const applicationId = String(application?.id ?? "").trim();

    if (!applicationId) {
      setLoadError(
        "Unable to open application details because its ID is missing. Refresh the applications list and try again."
      );
      return;
    }

    setLoadError("");
    setQuickViewApplication(null);

    navigate(
      `/app/applications/${encodeURIComponent(applicationId)}`,
      { state: { application } }
    );
  };

  /* ========================================================= */
  /* REFRESH */
  /* ========================================================= */

  const handleRefresh = () => loadApplications({ refresh: true });

  /* ========================================================= */
  /* RUN RESOURCE OPTIMIZATION */
  /* ========================================================= */

  const handleRunOptimization = async () => {
    if (!activeApplication?.id) {
      setOptimizationError(
        "Please select an application before running resource analysis."
      );
      return;
    }

    const parsedMachineId = Number(selectedMachineId);

    if (
      !Number.isInteger(parsedMachineId) ||
      parsedMachineId < 1
    ) {
      setOptimizationError(
        "Enter a valid Alibaba dataset machine ID (a positive whole number)."
      );
      return;
    }

    setIsRunningOptimization(true);
    setOptimizationError("");
    setOptimizationResult(null);
    setOptimizationHistory([]);

    try {
      // This endpoint uses an explicitly selected Alibaba sample workload.
      // It does not claim that the selected machine belongs to this application.
      const response = await API.post(
        `/optimization/applications/${encodeURIComponent(
          activeApplication.id
        )}/analyze-workload?machine_id=${parsedMachineId}`
      );

      const result = response?.data ?? response ?? null;
      setOptimizationResult(result);

      // Reflect the just-saved backend recommendation immediately in the
      // application cards and selected application view.
      const mlDecision = String(
        result?.recommendation?.action || ""
      ).toUpperCase();
      const mlTitle = String(
        result?.recommendation?.title || result?.recommendation?.message || ""
      );
      const resourceAnalysis = result?.resource_analysis ?? {};
      const savedLatestOptimization = {
        recommendation_id: result?.recommendation_id ?? null,
        action: mlDecision,
        title: mlTitle,
        priority: result?.recommendation?.priority ?? "",
        decision_source: resourceAnalysis?.decision_source ?? null,
        ml_analysis: resourceAnalysis?.ml_analysis ?? null,
        machine_id: resourceAnalysis?.machine_id ?? parsedMachineId,
        workload_source: resourceAnalysis?.data_source ?? "Alibaba Cluster Trace 2017",
        updated_at: result?.analysis_metadata?.analyzed_at ?? new Date().toISOString(),
      };

      setApplications((current) =>
        current.map((application) =>
          application.id === activeApplication.id
            ? {
                ...application,
                ...(["KEEP", "RESIZE", "SWITCH"].includes(mlDecision)
                  ? { decision: mlDecision }
                  : {}),
                recommendation: mlTitle || application.recommendation,
                latestOptimization: savedLatestOptimization,
              }
            : application
        )
      );

      if (["KEEP", "RESIZE", "SWITCH"].includes(mlDecision)) {
        setSelectedDecision(mlDecision);
      }

      try {
        const historyResponse = await getOptimizationHistory(activeApplication.id);
        const historyPayload = historyResponse?.data ?? historyResponse;
        const historyItems = Array.isArray(historyPayload)
          ? historyPayload
          : historyPayload?.recommendations ??
            historyPayload?.history ??
            historyPayload?.items ??
            historyPayload?.data ??
            [];
        setOptimizationHistory(Array.isArray(historyItems) ? historyItems : []);
      } catch (historyError) {
        // The current analysis is still useful even if history is unavailable.
        setOptimizationHistory([]);
      }

      setLastUpdated(
        new Date().toLocaleTimeString([], {
          hour: "2-digit",
          minute: "2-digit",
        })
      );
    } catch (error) {
      setOptimizationError(
        getOptimizationError(error) ||
          error.response?.data?.detail ||
          error.message ||
          "Unable to run forecast-aware resource analysis. Please try again."
      );
    } finally {
      setIsRunningOptimization(false);
    }
  };

  const readOptimizationValue = (...values) => {
    const value = values.find(
      (item) => item !== null && item !== undefined && item !== ""
    );
    return value === undefined ? null : value;
  };

  const formatOptimizationCurrency = (value) => {
    const parsed = toFiniteNumber(value, null);
    return parsed === null ? "Not available" : `$${parsed.toFixed(2)}`;
  };

  /* ========================================================= */
  /* RESET FILTERS */
  /* ========================================================= */

  const resetFilters = () => {
    setSearchQuery("");
    setProviderFilter("All");
    setSortBy("score");
  };

  /* ========================================================= */
  /* HELPERS */
  /* ========================================================= */

  const getScoreColor = (score) => {
    if (!Number.isFinite(score)) {
      return {
        text: "text-slate-500",
        bg: "bg-slate-50",
        border: "border-slate-200",
        progress: "bg-slate-300",
      };
    }

    if (score >= 85) {
      return {
        text: "text-emerald-600",
        bg: "bg-emerald-50",
        border: "border-emerald-100",
        progress: "bg-emerald-500",
      };
    }

    if (score >= 70) {
      return {
        text: "text-blue-600",
        bg: "bg-blue-50",
        border: "border-blue-100",
        progress: "bg-blue-500",
      };
    }

    return {
      text: "text-amber-600",
      bg: "bg-amber-50",
      border: "border-amber-100",
      progress: "bg-amber-500",
    };
  };

  const getStatusStyle = (status) => {
    if (status === "Healthy") {
      return "border-emerald-100 bg-emerald-50 text-emerald-600";
    }
    if (status === "Warning") {
      return "border-amber-100 bg-amber-50 text-amber-600";
    }
    return "border-slate-200 bg-slate-100 text-slate-500";
  };

  const getUsageStyle = (usage) => {
    if (usage === "High") {
      return "text-red-500";
    }

    if (usage === "Medium") {
      return "text-amber-500";
    }

    return "text-emerald-500";
  };

  const getResourceColor = (score) => {
    if (!Number.isFinite(score)) {
      return "bg-slate-300";
    }
    if (score >= 85) {
      return "bg-emerald-500";
    }

    if (score >= 70) {
      return "bg-blue-500";
    }

    return "bg-amber-500";
  };

  /* ========================================================= */
  /* DEFAULT DECISION */
  /* ========================================================= */

  const getDefaultDecision = (application) => {
    if (!application) {
      return "KEEP";
    }
    if (application.decision) return application.decision;

    if (
      application.usage === "High" ||
      application.usagePercent >= 85
    ) {
      return "RESIZE";
    }

    if (
      application.potentialSavings >= 25 &&
      application.optimizationScore < 90
    ) {
      return "SWITCH";
    }

    if (
      application.optimizationScore >= 90 &&
      application.healthScore >= 90
    ) {
      return "KEEP";
    }

    return "RESIZE";
  };

  /* ========================================================= */
  /* DECISION ENGINE */
  /* ========================================================= */

  const decisionData = useMemo(() => {
    if (!activeApplication) {
      return null;
    }

    // The application record does not contain verified post-change provider
    // pricing. Keep all three options as descriptive choices, but do not
    // manufacture savings, performance gains, compatibility, or risk scores.
    const currentCost = activeApplication.monthlyCost;

    return {
      KEEP: {
        label: "KEEP",
        title: "Keep Current Configuration",
        description:
          "Retain the current configuration. The ML-guided backend recommendation is shown after workload analysis.",
        cost: currentCost,
        savings: null,
        performance: null,
        compatibility: null,
        migrationRisk: "Not assessed",
        stability: "Not assessed",
        riskLevel: "Not assessed",
        icon: CircleCheck,
        service: `${activeApplication.provider} Current Stack`,
        reasons: [
          "This option preserves the existing cloud configuration.",
          "No provider-specific price comparison is available.",
          "No infrastructure change is performed automatically.",
        ],
      },

      RESIZE: {
        label: "RESIZE",
        title: "Resize Current Resources",
        description:
          "Consider changing resource allocation. The backend workload recommendation and safety checks determine whether resizing is appropriate.",
        cost: currentCost,
        savings: null,
        performance: null,
        compatibility: null,
        migrationRisk: "Not assessed",
        stability: "Not assessed",
        riskLevel: "Not assessed",
        icon: SlidersHorizontal,
        service: `${activeApplication.provider} Resource Configuration`,
        reasons: [
          "Resource resizing requires validation against workload demand.",
          "No verified post-resize provider price is available.",
          "No infrastructure change is performed automatically.",
        ],
      },

      SWITCH: {
        label: "SWITCH",
        title: "Switch Cloud Service",
        description:
          "Consider an alternative cloud service only after provider pricing, compatibility, and migration requirements have been validated.",
        cost: currentCost,
        savings: null,
        performance: null,
        compatibility: null,
        migrationRisk: "Not assessed",
        stability: "Not assessed",
        riskLevel: "Not assessed",
        icon: GitBranch,
        service: "Alternative provider (not evaluated)",
        reasons: [
          "A provider-to-provider price comparison has not been performed.",
          "Compatibility and migration effort have not been measured.",
          "No infrastructure change is performed automatically.",
        ],
      },
    };
  }, [activeApplication]);

  const backendDecisionAction = String(
    optimizationResult?.recommendation?.action || ""
  ).toUpperCase();

  const effectiveDecision = ["KEEP", "RESIZE", "SWITCH"].includes(
    backendDecisionAction
  )
    ? backendDecisionAction
    : selectedDecision;

  const currentDecision =
    decisionData?.[effectiveDecision] || null;

  /* ========================================================= */
  /* DECISION FACTORS */
  /* ========================================================= */

  const decisionFactors = activeApplication
    ? [
        {
          label: "Workload",
          value: activeApplication.usagePercent ?? 0,
          icon: Activity,
          description: "Observed utilization",
        },
        {
          label: "Health Score",
          value: activeApplication.healthScore,
          icon: Gauge,
          description: "Pending real health metrics",
        },
        {
          label: "Utilization Fit",
          value:
            activeApplication.optimizationScore,
          icon: Target,
          description: "Application-level fit score; not ML confidence",
        },
        {
          label: "Compatibility",
          value: null,
          icon: ShieldCheck,
          description: "Not measured by the current workload analysis",
        },
        {
          label: "Switching Stability",
          value: null,
          icon: Workflow,
          description:
            "Not measured; migration assessment required",
        },
      ]
    : [];

  /* ========================================================= */
  /* STATISTICS */
  /* ========================================================= */

  const statistics = [
    {
      title: "Cloud Applications",
      value: applications.length,
      description: "Connected applications",
      icon: Layers3,
      color: "blue",
    },
    {
      title: "Utilization Fit Score",
      value: averageOptimizationScore === null ? "Pending" : `${averageOptimizationScore}%`,
      description: "Application-level score; not ML confidence",
      icon: Target,
      color: "violet",
    },
    {
      title: "Potential Savings",
      value: `$${totalPotentialSavings.toFixed(0)}`,
      description: "Estimated monthly savings",
      icon: DollarSign,
      color: "emerald",
    },
    {
      title: "Cloud Services",
      value: totalServices,
      description: "Resources analyzed by AI",
      icon: Cpu,
      color: "orange",
    },
  ];

  const savedMlConfidence = toFiniteNumber(
    activeApplication?.latestOptimization?.ml_analysis?.prediction?.confidence,
    null
  );

  const displayedMlConfidence = toFiniteNumber(
    optimizationResult?.resource_analysis?.ml_analysis?.prediction?.confidence,
    savedMlConfidence
  );

  const getStatColor = (color) => {
    const colors = {
      blue: "bg-blue-50 text-blue-600",
      violet:
        "bg-violet-50 text-violet-600",
      emerald:
        "bg-emerald-50 text-emerald-600",
      orange:
        "bg-orange-50 text-orange-600",
    };

    return colors[color];
  };

  /* ========================================================= */
  /* RENDER */
  /* ========================================================= */

  return (
    <div className="min-h-screen bg-slate-100">
      {/* ===================================================== */}
      {/* HERO */}
      {/* ===================================================== */}

      <section className="relative overflow-hidden bg-gradient-to-br from-slate-950 via-blue-950 to-violet-950">
        <div className="absolute -right-20 -top-20 h-96 w-96 rounded-full bg-blue-500/20 blur-3xl" />

        <div className="absolute bottom-0 left-1/3 h-72 w-72 rounded-full bg-violet-500/20 blur-3xl" />

        <div className="absolute left-0 top-1/2 h-64 w-64 rounded-full bg-cyan-500/10 blur-3xl" />

        <div className="relative px-8 py-10">
          <div className="flex flex-col gap-6 lg:flex-row lg:items-center lg:justify-between">
            <div>
              <div className="flex flex-wrap items-center gap-3">
                <span className="flex items-center gap-2 rounded-full border border-violet-400/20 bg-violet-500/10 px-4 py-2 text-xs font-bold uppercase tracking-wider text-violet-200">
                  <BrainCircuit size={15} />
                  CloudMorph AI Engine
                </span>

                <span className="flex items-center gap-2 text-xs font-semibold text-emerald-300">
                  <span className="h-2 w-2 animate-pulse rounded-full bg-emerald-400" />
                  ML + Forecast Policy
                </span>
              </div>

              <h1 className="mt-6 text-3xl font-bold tracking-tight text-white md:text-5xl">
                AI Cloud Optimizer
              </h1>

              <p className="mt-4 max-w-2xl text-sm leading-relaxed text-slate-300 md:text-base">
                Intelligent cloud infrastructure analysis
                powered by AI. Discover optimization
                opportunities and determine whether
                infrastructure should KEEP, RESIZE, or
                SWITCH.
              </p>

              <div className="mt-5 flex items-center gap-2 text-xs text-slate-400">
                <Clock3 size={15} />

                Last AI analysis updated at{" "}
                <span className="font-semibold text-slate-200">
                  {lastUpdated}
                </span>
              </div>
            </div>

            <button
              type="button"
              onClick={handleRefresh}
              disabled={isRefreshing}
              className="flex items-center justify-center gap-2 rounded-xl bg-white px-5 py-3.5 text-sm font-bold text-slate-900 shadow-lg transition hover:bg-slate-100 disabled:cursor-not-allowed disabled:opacity-70"
            >
              <RefreshCw
                size={18}
                className={
                  isRefreshing
                    ? "animate-spin"
                    : ""
                }
              />

              {isRefreshing
                ? "Refreshing AI..."
                : "Refresh AI Analysis"}
            </button>
          </div>

          <div className="mt-10 grid grid-cols-1 gap-4 md:grid-cols-3">
            <StatusCard
              icon={CheckCircle2}
              iconStyle="bg-emerald-500/15 text-emerald-300"
              label="AI Analysis Status"
              value={loadError ? "Data Unavailable" : isLoading ? "Loading Applications" : "Application Data Loaded"}
            />

            <StatusCard
              icon={Activity}
              iconStyle="bg-blue-500/15 text-blue-300"
              label="Resources Monitored"
              value={`${totalServices} Cloud Services`}
            />

            <StatusCard
              icon={ShieldCheck}
              iconStyle="bg-violet-500/15 text-violet-300"
              label="AI Confidence"
              value={
                displayedMlConfidence !== null
                  ? `${(
                      displayedMlConfidence <= 1
                        ? displayedMlConfidence * 100
                        : displayedMlConfidence
                    ).toFixed(2)}% model confidence`
                  : "Run workload analysis"
              }
            />
          </div>
        </div>
      </section>

      {/* ===================================================== */}
      {/* MAIN */}
      {/* ===================================================== */}

      <main className="p-8">
        <div className="mb-6 flex flex-col gap-2">
          <div className="flex items-center gap-2">
            <BarChart3
              size={22}
              className="text-blue-600"
            />

            <h2 className="text-xl font-bold text-slate-800">
              Optimization Overview
            </h2>
          </div>

          <p className="text-sm text-slate-500">
            AI-generated insights across your cloud
            infrastructure.
          </p>
        </div>

        {loadError && (
          <div className="mb-5 rounded-xl border border-red-200 bg-red-50 p-4 text-sm text-red-700">
            Could not refresh application data: {String(loadError)}
            <button type="button" onClick={() => loadApplications()} className="ml-3 font-bold underline">Retry</button>
          </div>
        )}
        {isLoading && <p className="mb-5 text-sm text-slate-500">Loading your applications...</p>}
        {!isLoading && !loadError && applications.length === 0 && (
          <div className="mb-5 rounded-xl border border-slate-200 bg-white p-5 text-sm text-slate-600">
            No applications found. Add an application from the Applications page to see its saved optimization data.
          </div>
        )}
        {/* =================================================== */}
        {/* STATISTICS */}
        {/* =================================================== */}

        <div className="grid grid-cols-1 gap-5 md:grid-cols-2 xl:grid-cols-4">
          {statistics.map((stat) => {
            const Icon = stat.icon;

            return (
              <div
                key={stat.title}
                className="group rounded-2xl border border-slate-200 bg-white p-6 shadow-sm transition duration-300 hover:-translate-y-1 hover:border-blue-200 hover:shadow-xl"
              >
                <div className="flex items-start justify-between">
                  <div
                    className={`flex h-12 w-12 items-center justify-center rounded-xl ${getStatColor(
                      stat.color
                    )}`}
                  >
                    <Icon size={23} />
                  </div>

                  <ArrowUpRight
                    size={18}
                    className="text-slate-300 transition group-hover:text-blue-500"
                  />
                </div>

                <p className="mt-6 text-sm font-medium text-slate-500">
                  {stat.title}
                </p>

                <p className="mt-2 text-3xl font-bold text-slate-800">
                  {stat.value}
                </p>

                <p className="mt-2 text-xs text-slate-400">
                  {stat.description}
                </p>
              </div>
            );
          })}
        </div>

        {/* =================================================== */}
        {/* QUICK INSIGHTS */}
        {/* =================================================== */}

        <div className="mt-6 grid grid-cols-1 gap-5 md:grid-cols-3">
          <QuickInsight
            icon={CircleDollarSign}
            iconColor="text-blue-600"
            title="Monthly Cloud Spending"
            value={`$${totalMonthlyCost.toFixed(2)}`}
            description="Current infrastructure cost"
          />

          <QuickInsight
            icon={ShieldCheck}
            iconColor="text-violet-600"
            title="Average Health Score"
            value={averageHealthScore === null ? "Pending" : `${averageHealthScore}%`}
            description="Requires uptime, latency, and error-rate metrics"
          />

          <QuickInsight
            icon={CheckCircle2}
            iconColor="text-emerald-600"
            title="Healthy Applications"
            value={`${healthyApplications}/${applications.length}`}
            description="Applications performing normally"
          />
        </div>

        {/* =================================================== */}
        {/* AI DECISION ENGINE */}
        {/* =================================================== */}

        {activeApplication && decisionData && <section className="mt-10 overflow-hidden rounded-3xl border border-slate-200 bg-white shadow-sm">
          <div className="border-b border-slate-200 bg-gradient-to-r from-slate-50 via-white to-blue-50 p-7">
            <div className="flex min-w-0 flex-col gap-6">
              <div>
                <div className="flex items-center gap-2">
                  <div className="rounded-xl bg-gradient-to-br from-blue-600 to-violet-600 p-2.5 text-white">
                    <BrainCircuit size={21} />
                  </div>

                  <div>
                    <p className="text-xs font-bold uppercase tracking-[0.18em] text-violet-600">
                      Decision Engine
                    </p>

                    <h2 className="mt-1 text-2xl font-bold text-slate-800">
                      KEEP / RESIZE / SWITCH
                    </h2>
                  </div>
                </div>

                <p className="mt-3 max-w-3xl text-sm leading-relaxed text-slate-500">
                  CloudMorph evaluates workload, cost,
                  performance, compatibility, migration
                  risk, and switching stability before
                  producing an infrastructure recommendation.
                </p>
              </div>

              <div className="grid w-full min-w-0 grid-cols-1 gap-3 sm:grid-cols-2">
                <button
                  type="button"
                  onClick={handleRunOptimization}
                  disabled={isRunningOptimization}
                  className="flex shrink-0 items-center justify-center gap-2 rounded-xl bg-slate-900 px-5 py-3.5 text-sm font-bold text-white shadow-lg transition hover:bg-blue-600 disabled:cursor-not-allowed disabled:opacity-70"
                >
                  <Sparkles
                    size={17}
                    className={isRunningOptimization ? "animate-pulse" : ""}
                  />
                  {isRunningOptimization ? "Running ML Decision..." : "Run AI Decision"}
                </button>

                <div className="grid min-w-0 grid-cols-1 gap-3 sm:grid-cols-[minmax(105px,0.55fr)_minmax(0,1.45fr)]">
                  <div className="min-w-0">
                    <label
                      htmlFor="workload-machine-id"
                      className="mb-1.5 block text-xs font-semibold text-slate-500"
                    >
                      Alibaba sample machine ID
                    </label>
                    <input
                      id="workload-machine-id"
                      type="number"
                      min="1"
                      step="1"
                      value={selectedMachineId}
                      onChange={(event) => setSelectedMachineId(event.target.value)}
                      disabled={isRunningOptimization}
                      className="w-full min-w-0 rounded-xl border border-slate-200 bg-white px-3 py-3 text-sm font-semibold text-slate-700 outline-none transition focus:border-blue-400 focus:ring-4 focus:ring-blue-100"
                      placeholder="e.g. 1"
                    />
                  </div>

                  <div className="flex min-w-0 items-center rounded-xl border border-blue-100 bg-blue-50 px-3 py-3 text-xs leading-relaxed text-blue-800">
                    This machine ID is used by the ML workload classifier and forecast-aware safety checks. The AI Decision button runs this analysis.
                  </div>
                </div>
              </div>
            </div>
          </div>

          <div className="p-7">
            {/* WORKFLOW */}

            <div className="mb-8 grid grid-cols-1 gap-3 md:grid-cols-4">
              <WorkflowStep
                number="01"
                title="Monitor"
                description="Workload & usage"
                icon={Activity}
              />

              <WorkflowStep
                number="02"
                title="Evaluate"
                description="Cost & performance"
                icon={BarChart3}
              />

              <WorkflowStep
                number="03"
                title="Decide"
                description="AI decision engine"
                icon={BrainCircuit}
              />

              <WorkflowStep
                number="04"
                title="Optimize"
                description="Action recommendation"
                icon={Zap}
              />
            </div>

            {/* ACTIVE APPLICATION */}

            <div className="rounded-2xl border border-slate-200 bg-slate-50 p-5">
              <div className="flex flex-col gap-5 lg:flex-row lg:items-center lg:justify-between">
                <div className="flex items-center gap-4">
                  <div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-gradient-to-br from-blue-600 to-violet-600 text-white shadow-lg">
                    <BrainCircuit size={26} />
                  </div>

                  <div className="min-w-0 flex-1">
                    <p className="text-xs font-bold uppercase tracking-wider text-slate-400">
                      Application Under Evaluation
                    </p>

                    <h3 className="mt-1 text-xl font-bold text-slate-800">
                      {activeApplication.name}
                    </h3>

                    <p className="mt-1 text-xs text-slate-500">
                      {activeApplication.provider} •{" "}
                      {activeApplication.category} •{" "}
                      {activeApplication.services} services
                    </p>

                    {applications.length > 1 && (
                      <div className="mt-3 max-w-sm">
                        <label
                          htmlFor="active-application-selector"
                          className="mb-1.5 block text-xs font-semibold text-slate-500"
                        >
                          Switch application
                        </label>
                        <select
                          id="active-application-selector"
                          value={activeApplication.id}
                          onChange={(event) => {
                            const nextApplication = applications.find(
                              (application) => application.id === event.target.value
                            );
                            if (nextApplication) {
                              setSelectedApplication(nextApplication);
                              setSelectedDecision(nextApplication.decision || "KEEP");
                              setLoadError("");
                            }
                          }}
                          className="w-full rounded-xl border border-slate-200 bg-white px-3 py-2.5 text-sm font-semibold text-slate-700 outline-none transition focus:border-blue-400 focus:ring-4 focus:ring-blue-100"
                        >
                          {applications.map((application) => (
                            <option key={application.id} value={application.id}>
                              {application.name} ({application.provider})
                            </option>
                          ))}
                        </select>
                      </div>
                    )}
                  </div>
                </div>

                <div className="grid grid-cols-3 gap-3">
                  <DecisionMiniStat
                    label="Cost"
                    value={`$${activeApplication.monthlyCost.toFixed(
                      2
                    )}`}
                  />

                  <DecisionMiniStat
                    label="Usage"
                    value={formatScore(activeApplication.usagePercent)}
                  />

                  <DecisionMiniStat
                    label="Score"
                    value={formatScore(activeApplication.optimizationScore)}
                  />
                </div>
              </div>
            </div>

            {/* DECISION OPTIONS */}

            <div className="mt-7">
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="text-lg font-bold text-slate-800">
                    AI Decision Options
                  </h3>

                  <p className="mt-1 text-xs text-slate-400">
                    After analysis, the ML backend recommendation is highlighted. Other options are informational only.
                  </p>
                </div>

                <span className="hidden rounded-full bg-violet-50 px-3 py-1.5 text-[10px] font-bold uppercase tracking-wider text-violet-600 sm:block">
                  Decision Preview
                </span>
              </div>

              <div className="mt-5 grid grid-cols-1 gap-4 md:grid-cols-3">
                {Object.entries(decisionData).map(
                  ([key, decision]) => {
                    const Icon = decision.icon;
                    const isSelected =
                      selectedDecision === key;

                    const cardColors = {
                      KEEP: isSelected
                        ? "border-emerald-400 bg-emerald-50 shadow-lg shadow-emerald-100"
                        : "border-slate-200 bg-white hover:border-emerald-200",
                      RESIZE: isSelected
                        ? "border-blue-400 bg-blue-50 shadow-lg shadow-blue-100"
                        : "border-slate-200 bg-white hover:border-blue-200",
                      SWITCH: isSelected
                        ? "border-violet-400 bg-violet-50 shadow-lg shadow-violet-100"
                        : "border-slate-200 bg-white hover:border-violet-200",
                    };

                    const iconColors = {
                      KEEP:
                        "bg-emerald-100 text-emerald-600",
                      RESIZE:
                        "bg-blue-100 text-blue-600",
                      SWITCH:
                        "bg-violet-100 text-violet-600",
                    };

                    return (
                      <button
                        key={key}
                        type="button"
                        onClick={() => {
                          if (!optimizationResult) setSelectedDecision(key);
                        }}
                        disabled={Boolean(optimizationResult)}
                        className={`group relative overflow-hidden rounded-2xl border p-5 text-left transition duration-300 ${optimizationResult ? "cursor-default" : "hover:-translate-y-1"} ${cardColors[key]}`}
                      >
                        {isSelected && (
                          <div className="absolute right-4 top-4">
                            <CheckCircle2
                              size={19}
                              className={
                                key === "KEEP"
                                  ? "text-emerald-500"
                                  : key === "RESIZE"
                                  ? "text-blue-500"
                                  : "text-violet-500"
                              }
                            />
                          </div>
                        )}

                        <div
                          className={`flex h-12 w-12 items-center justify-center rounded-xl ${iconColors[key]}`}
                        >
                          <Icon size={23} />
                        </div>

                        <p className="mt-5 text-xs font-bold uppercase tracking-[0.18em] text-slate-400">
                          {decision.label}
                        </p>

                        <h4 className="mt-2 text-lg font-bold text-slate-800">
                          {decision.title}
                        </h4>

                        <p className="mt-2 text-xs leading-relaxed text-slate-500">
                          {decision.description}
                        </p>

                        <div className="mt-5 flex items-center justify-between border-t border-slate-200/70 pt-4">
                          <span className="text-xs text-slate-400">
                            Current cost reference
                          </span>

                          <span className="font-bold text-slate-800">
                            ${decision.cost.toFixed(2)}
                          </span>
                        </div>
                      </button>
                    );
                  }
                )}
              </div>
            </div>

            {/* SELECTED DECISION */}

            {currentDecision && (
              <div className="mt-7 grid grid-cols-1 gap-6 xl:grid-cols-3">
                {/* RESULT */}

                <div className="overflow-hidden rounded-2xl border border-slate-200 bg-slate-50 xl:col-span-2">
                  <div
                    className={`border-b p-6 ${
                      selectedDecision === "KEEP"
                        ? "border-emerald-100 bg-emerald-50"
                        : selectedDecision ===
                          "RESIZE"
                        ? "border-blue-100 bg-blue-50"
                        : "border-violet-100 bg-violet-50"
                    }`}
                  >
                    <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
                      <div className="flex items-center gap-4">
                        <div
                          className={`rounded-2xl p-3 ${
                            selectedDecision ===
                            "KEEP"
                              ? "bg-emerald-500 text-white"
                              : selectedDecision ===
                                "RESIZE"
                              ? "bg-blue-500 text-white"
                              : "bg-violet-500 text-white"
                          }`}
                        >
                          <currentDecision.icon
                            size={25}
                          />
                        </div>

                        <div>
                          <p className="text-xs font-bold uppercase tracking-wider text-slate-400">
                            Recommended Action
                          </p>

                          <h3 className="mt-1 text-2xl font-bold text-slate-800">
                            {currentDecision.label}
                          </h3>
                        </div>
                      </div>

                      <span className="rounded-full bg-white px-4 py-2 text-xs font-bold text-slate-600 shadow-sm">
                        {optimizationResult
                          ? "ML-guided backend recommendation"
                          : "Illustrative option — run AI analysis"}
                      </span>
                    </div>
                  </div>

                  <div className="p-6">
                    <p className="text-sm leading-relaxed text-slate-600">
                      {currentDecision.description}
                    </p>

                    {/* CURRENT → RECOMMENDED */}

                    <div className="mt-6 grid grid-cols-1 gap-3 md:grid-cols-3">
                      <InfrastructureBox
                        label="Current"
                        value={`${activeApplication.provider} Current Stack`}
                        icon={Cloud}
                      />

                      <div className="hidden items-center justify-center md:flex">
                        <ArrowRight className="text-slate-300" />
                      </div>

                      <InfrastructureBox
                        label="Recommended"
                        value={currentDecision.service}
                        icon={
                          selectedDecision ===
                          "SWITCH"
                            ? GitBranch
                            : selectedDecision ===
                              "RESIZE"
                            ? SlidersHorizontal
                            : CheckCircle2
                        }
                      />
                    </div>

                    {/* IMPACT */}

                    <div className="mt-6 grid grid-cols-2 gap-3 lg:grid-cols-4">
                      <ImpactMetric
                        icon={DollarSign}
                        label="Monthly Cost"
                        value={`$${currentDecision.cost.toFixed(
                          2
                        )}`}
                        helper="Current cost only; no repricing"
                        positive={false}
                      />

                      <ImpactMetric
                        icon={TrendingDown}
                        label="Savings"
                        value="Not calculated"
                        helper="Provider pricing unavailable"
                        positive={false}
                      />

                      <ImpactMetric
                        icon={TrendingUp}
                        label="Performance"
                        value="Not measured"
                        helper="No performance benchmark"
                        positive={false}
                      />

                      <ImpactMetric
                        icon={ShieldAlert}
                        label="Migration Risk"
                        value={
                          currentDecision.migrationRisk
                        }
                        helper="Requires migration assessment"
                        positive={false}
                      />
                    </div>

                    {/* REASONING */}

                    <div className="mt-7">
                      <div className="flex items-center gap-2">
                        <Lightbulb
                          size={18}
                          className="text-violet-600"
                        />

                        <h4 className="font-bold text-slate-800">
                          Decision Reasoning
                        </h4>
                      </div>

                      <div className="mt-4 grid grid-cols-1 gap-3 md:grid-cols-2">
                        {currentDecision.reasons.map(
                          (reason, index) => (
                            <div
                              key={reason}
                              className="flex items-start gap-3 rounded-xl border border-slate-100 bg-white p-4"
                            >
                              <div className="mt-0.5 flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-violet-50 text-violet-600">
                                <span className="text-[10px] font-bold">
                                  {index + 1}
                                </span>
                              </div>

                              <p className="text-xs leading-relaxed text-slate-500">
                                {reason}
                              </p>
                            </div>
                          )
                        )}
                      </div>
                    </div>
                  </div>
                </div>

                {/* FACTORS */}

                <div className="rounded-2xl border border-slate-200 bg-white p-6">
                  <div className="flex items-center gap-2">
                    <Target
                      size={20}
                      className="text-blue-600"
                    />

                    <div>
                      <h3 className="font-bold text-slate-800">
                        Decision Factors
                      </h3>

                      <p className="mt-1 text-[10px] text-slate-400">
                        Inputs considered by the decision engine
                      </p>
                    </div>
                  </div>

                  <div className="mt-6 space-y-5">
                    {decisionFactors.map(
                      (factor) => {
                        const Icon = factor.icon;

                        return (
                          <div
                            key={factor.label}
                          >
                            <div className="flex items-center justify-between">
                              <div className="flex items-center gap-2">
                                <Icon
                                  size={15}
                                  className="text-slate-400"
                                />

                                <span className="text-xs font-semibold text-slate-600">
                                  {factor.label}
                                </span>
                              </div>

                              <span className="text-xs font-bold text-slate-700">
                                {Number.isFinite(factor.value)
                                  ? `${Math.round(factor.value)}%`
                                  : "Pending"}
                              </span>
                            </div>

                            {Number.isFinite(factor.value) && (
                              <div className="mt-2 h-2 overflow-hidden rounded-full bg-slate-100">
                                <div
                                  className="h-full rounded-full bg-gradient-to-r from-blue-500 to-violet-500 transition-all duration-700"
                                  style={{
                                    width: `${Math.max(0, Math.min(100, factor.value))}%`,
                                  }}
                                />
                              </div>
                            )}

                            <p className="mt-1 text-[10px] text-slate-400">
                              {factor.description}
                            </p>
                          </div>
                        );
                      }
                    )}
                  </div>

                  <div className="mt-7 rounded-xl bg-slate-50 p-4">
                    <div className="flex items-start gap-3">
                      <ShieldCheck
                        size={19}
                        className="mt-0.5 text-emerald-500"
                      />

                      <div>
                        <p className="text-xs font-bold text-slate-700">
                          Migration Assessment
                        </p>

                        <p className="mt-1 text-[10px] leading-relaxed text-slate-500">
                          Compatibility and switching stability
                          are considered so that cost
                          optimization does not create
                          unnecessary operational risk.
                        </p>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        </section>}

        {/* =================================================== */}
        {/* RESOURCE OPTIMIZATION RESULTS */}
        {/* =================================================== */}

        {(optimizationError || optimizationResult || optimizationHistory.length > 0) && (
          <section className="mt-8 rounded-3xl border border-slate-200 bg-white p-7 shadow-sm">
            <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
              <div>
                <div className="flex items-center gap-2">
                  <Activity size={22} className="text-blue-600" />
                  <h2 className="text-xl font-bold text-slate-800">
                    Resource Analysis Results
                  </h2>
                </div>
                <p className="mt-2 text-sm text-slate-500">
                  Results returned by the CloudMorph optimization API for{" "}
                  {activeApplication?.name || "the selected application"}.
                </p>
              </div>
              {optimizationResult && (
                <span className="rounded-full bg-blue-50 px-3 py-1.5 text-xs font-bold text-blue-700">
                  Forecast-aware backend analysis
                </span>
              )}
            </div>

            {optimizationError && (
              <div className="mt-5 rounded-xl border border-red-200 bg-red-50 p-4 text-sm text-red-700">
                {optimizationError}
              </div>
            )}

            {optimizationResult && (() => {
              const result = optimizationResult;
              const recommendation = result.recommendation ?? result.result?.recommendation ?? {};
              const applicationInfo = result.application ?? result.result?.application ?? {};
              const resourceAnalysis = result.resource_analysis ?? result.resourceAnalysis ?? result.result?.resource_analysis ?? {};
              const memoryRiskAnalysis =
                resourceAnalysis.memory_risk_analysis ??
                resourceAnalysis.memoryRiskAnalysis ??
                result.memory_risk_analysis ??
                result.memoryRiskAnalysis ??
                result.result?.resource_analysis?.memory_risk_analysis ??
                {};
              const workloadAnalysis = resourceAnalysis.workload_analysis ?? {};
              const resourceStatistics = workloadAnalysis.resource_statistics ?? {};
              const workloadTrends = workloadAnalysis.workload_trends ?? {};
              const workloadForecasts = workloadAnalysis.forecasts ?? result.forecast_analysis?.forecasts ?? {};
              const resourcePressure = resourceAnalysis.resource_pressure ?? {};
              const utilization = resourceAnalysis.utilization ?? result.utilization ?? {};
              const costAnalysis = result.cost_analysis ?? result.costAnalysis ?? result.result?.cost_analysis ?? {};
              const workloadSource = resourceAnalysis.data_source ?? result.analysis_metadata?.workload_source;
              const workloadMachineId = resourceAnalysis.machine_id;
              const forecastMethod = result.forecast_analysis?.forecast_method ?? workloadAnalysis.input_metadata?.forecast_method;
              const mlAnalysis = resourceAnalysis.ml_analysis ?? {};
              const mlPrediction = mlAnalysis.prediction ?? {};
              const mlProbabilities = mlPrediction.class_probabilities ?? {};
              const mlConfidence = toFiniteNumber(mlPrediction.confidence, null);
              const mlConfidencePercent =
                mlConfidence === null
                  ? null
                  : mlConfidence <= 1
                    ? mlConfidence * 100
                    : mlConfidence;
              const decisionSource =
                resourceAnalysis.decision_source ??
                result.analysis_metadata?.decision_source ??
                "Not available";
              const mlClassifierUsed =
                result.analysis_metadata?.ml_classifier_used ??
                mlAnalysis.status === "success";
              const mlFallbackUsed =
                result.analysis_metadata?.ml_fallback_used ?? false;
              const formatProbability = (value) => {
                const parsed = toFiniteNumber(value, null);
                if (parsed === null) return "Not available";
                return `${(parsed <= 1 ? parsed * 100 : parsed).toFixed(2)}%`;
              };
              const formatPercentValue = (value) => {
                const parsed = toFiniteNumber(value, null);
                return parsed === null ? "Not available" : `${parsed.toFixed(2)}%`;
              };
              const action = readOptimizationValue(
                recommendation.action,
                recommendation.decision,
                result.action,
                result.decision
              );
              const title = readOptimizationValue(
                recommendation.title,
                recommendation.message,
                result.title,
                result.message
              );
              const reasons = readOptimizationValue(
                recommendation.reasons,
                result.reasons
              );
              const monthlySavings = readOptimizationValue(
                costAnalysis.estimated_monthly_savings,
                costAnalysis.monthly_savings,
                costAnalysis.estimatedMonthlySavings,
                result.estimated_monthly_savings,
                result.monthly_savings
              );
              const annualSavings = readOptimizationValue(
                costAnalysis.estimated_annual_savings,
                costAnalysis.annual_savings,
                costAnalysis.estimatedAnnualSavings,
                result.estimated_annual_savings,
                result.annual_savings
              );
              const utilizationEntries = Object.entries(utilization).filter(
                ([, value]) => toFiniteNumber(value, null) !== null
              );

              return (
                <div className="mt-6">
                  <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
                    <div className="rounded-2xl border border-slate-100 bg-slate-50 p-5">
                      <p className="text-xs font-semibold uppercase tracking-wider text-slate-400">
                        Recommended Action
                      </p>
                      <p className="mt-2 text-2xl font-bold text-slate-800">
                        {action ? String(action).toUpperCase() : "Not available"}
                      </p>
                      <p className="mt-1 text-sm text-slate-500">
                        {title || "No recommendation title returned"}
                      </p>
                    </div>

                    <div className="rounded-2xl border border-slate-100 bg-slate-50 p-5">
                      <p className="text-xs font-semibold uppercase tracking-wider text-slate-400">
                        Estimated Monthly Savings
                      </p>
                      <p className="mt-2 text-2xl font-bold text-emerald-600">
                        {formatOptimizationCurrency(monthlySavings)}
                      </p>
                      <p className="mt-1 text-xs text-slate-400">
                        Based on the API response
                      </p>
                    </div>

                    <div className="rounded-2xl border border-slate-100 bg-slate-50 p-5">
                      <p className="text-xs font-semibold uppercase tracking-wider text-slate-400">
                        Estimated Annual Savings
                      </p>
                      <p className="mt-2 text-2xl font-bold text-emerald-600">
                        {formatOptimizationCurrency(
                          annualSavings ??
                            (toFiniteNumber(monthlySavings, null) === null
                              ? null
                              : toFiniteNumber(monthlySavings) * 12)
                        )}
                      </p>
                      <p className="mt-1 text-xs text-slate-400">
                        Annualized estimate
                      </p>
                    </div>
                  </div>

                  {mlAnalysis.status === "success" && (
                    <div className="mt-6 rounded-2xl border border-violet-100 bg-gradient-to-br from-violet-50 via-white to-blue-50 p-5">
                      <div className="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
                        <div>
                          <div className="flex items-center gap-2">
                            <BrainCircuit size={19} className="text-violet-600" />
                            <h3 className="font-bold text-slate-800">
                              ML Workload Prediction
                            </h3>
                            <span className="rounded-full bg-emerald-100 px-2.5 py-1 text-[10px] font-bold uppercase tracking-wide text-emerald-700">
                              Classifier active
                            </span>
                          </div>
                          <p className="mt-1 text-xs text-slate-500">
                            {mlPrediction.model || "ML classifier"}
                            {mlPrediction.model_version
                              ? ` • Version ${mlPrediction.model_version}`
                              : ""}
                            {mlPrediction.prediction_horizon_minutes
                              ? ` • ${mlPrediction.prediction_horizon_minutes}-minute horizon`
                              : ""}
                          </p>
                        </div>
                        <div className="rounded-xl border border-violet-100 bg-white px-4 py-3">
                          <p className="text-[10px] font-bold uppercase tracking-wider text-slate-400">
                            Predicted condition
                          </p>
                          <p className="mt-1 text-xl font-extrabold text-violet-700">
                            {mlPrediction.predicted_condition || "Not available"}
                          </p>
                        </div>
                      </div>

                      <div className="mt-5 grid grid-cols-1 gap-4 md:grid-cols-3">
                        <div className="rounded-xl border border-slate-100 bg-white p-4">
                          <p className="text-xs font-semibold uppercase tracking-wider text-slate-400">
                            Model confidence
                          </p>
                          <p className="mt-2 text-3xl font-bold text-slate-800">
                            {mlConfidencePercent === null
                              ? "Not available"
                              : `${mlConfidencePercent.toFixed(2)}%`}
                          </p>
                          {mlConfidencePercent !== null && (
                            <div className="mt-3 h-2 overflow-hidden rounded-full bg-slate-100">
                              <div
                                className="h-full rounded-full bg-violet-500"
                                style={{
                                  width: `${Math.max(0, Math.min(100, mlConfidencePercent))}%`,
                                }}
                              />
                            </div>
                          )}
                          <p className="mt-2 text-[11px] text-slate-400">
                            {mlPrediction.feature_count ?? "—"} engineered features
                            {" • "}
                            {mlPrediction.input_history_minutes ?? "—"} minutes of history
                          </p>
                        </div>

                        <div className="rounded-xl border border-slate-100 bg-white p-4 md:col-span-2">
                          <p className="text-xs font-semibold uppercase tracking-wider text-slate-400">
                            Class probabilities
                          </p>
                          <div className="mt-3 space-y-3">
                            {[
                              { key: "HIGH_DEMAND", label: "High demand", color: "bg-rose-500" },
                              { key: "NORMAL_DEMAND", label: "Normal demand", color: "bg-emerald-500" },
                              { key: "LOW_DEMAND", label: "Low demand", color: "bg-amber-500" },
                            ].map((item) => {
                              const probability = toFiniteNumber(
                                mlProbabilities[item.key],
                                null
                              );
                              const percentage =
                                probability === null
                                  ? null
                                  : probability <= 1
                                    ? probability * 100
                                    : probability;
                              return (
                                <div key={item.key}>
                                  <div className="mb-1 flex items-center justify-between gap-3 text-xs">
                                    <span className="font-medium text-slate-600">
                                      {item.label}
                                    </span>
                                    <span className="font-bold tabular-nums text-slate-700">
                                      {formatProbability(probability)}
                                    </span>
                                  </div>
                                  <div className="h-2 overflow-hidden rounded-full bg-slate-100">
                                    {percentage !== null && (
                                      <div
                                        className={`h-full rounded-full ${item.color}`}
                                        style={{
                                          width: `${Math.max(0, Math.min(100, percentage))}%`,
                                        }}
                                      />
                                    )}
                                  </div>
                                </div>
                              );
                            })}
                          </div>
                        </div>
                      </div>

                      <div className="mt-4 grid grid-cols-1 gap-3 sm:grid-cols-3">
                        <div className="rounded-xl bg-white/80 p-3">
                          <p className="text-[10px] font-semibold uppercase text-slate-400">
                            Decision source
                          </p>
                          <p className="mt-1 break-words text-xs font-bold text-slate-700">
                            {String(decisionSource).replaceAll("_", " ")}
                          </p>
                        </div>
                        <div className="rounded-xl bg-white/80 p-3">
                          <p className="text-[10px] font-semibold uppercase text-slate-400">
                            Classifier status
                          </p>
                          <p className="mt-1 text-xs font-bold text-slate-700">
                            {mlClassifierUsed ? "Used" : "Not used"}
                            {mlFallbackUsed ? " • Safety fallback used" : ""}
                          </p>
                        </div>
                        <div className="rounded-xl bg-white/80 p-3">
                          <p className="text-[10px] font-semibold uppercase text-slate-400">
                            Confidence threshold
                          </p>
                          <p className="mt-1 text-xs font-bold text-slate-700">
                            {formatProbability(mlAnalysis.minimum_confidence_required)}
                          </p>
                        </div>
                      </div>

                      {mlAnalysis.fallback_reason && (
                        <p className="mt-3 rounded-lg border border-amber-200 bg-amber-50 p-3 text-xs text-amber-800">
                          Fallback reason: {mlAnalysis.fallback_reason}
                        </p>
                      )}

                      <p className="mt-4 text-[11px] leading-relaxed text-slate-500">
                        This is a workload-condition prediction from the selected Alibaba Cluster Trace sample.
                        It is advisory and does not represent live telemetry from this application's cloud provider.
                      </p>
                    </div>
                  )}

                  {memoryRiskAnalysis.status === "success" && (
                    <div className="mt-6 rounded-2xl border border-amber-200 bg-gradient-to-br from-amber-50 via-white to-orange-50 p-5">
                      {(() => {
                        const rawRiskProbability = toFiniteNumber(
                          memoryRiskAnalysis.risk_probability_percent,
                          null
                        );
                        const rawRiskScore = toFiniteNumber(
                          memoryRiskAnalysis.risk_probability,
                          null
                        );
                        const riskProbability =
                          rawRiskProbability !== null
                            ? rawRiskProbability
                            : rawRiskScore !== null
                              ? rawRiskScore * 100
                              : null;
                        const threshold = toFiniteNumber(
                          memoryRiskAnalysis.threshold,
                          null
                        );
                        const isHighRisk =
                          memoryRiskAnalysis.high_memory_risk === true;

                        return (
                          <>
                            <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
                              <div>
                                <div className="flex flex-wrap items-center gap-2">
                                  <BrainCircuit
                                    size={19}
                                    className="text-amber-600"
                                  />
                                  <h3 className="font-bold text-slate-800">
                                    ML Memory Risk Analysis
                                  </h3>
                                  <span className="rounded-full bg-amber-100 px-2.5 py-1 text-[10px] font-bold uppercase tracking-wide text-amber-800">
                                    Experimental ML model
                                  </span>
                                </div>
                                <p className="mt-1 text-xs text-slate-500">
                                  {memoryRiskAnalysis.model || "ExtraTreesClassifier"}
                                  {memoryRiskAnalysis.model_version
                                    ? ` • Version ${memoryRiskAnalysis.model_version}`
                                    : ""}
                                </p>
                              </div>

                              <span
                                className={`rounded-xl px-4 py-3 text-sm font-bold ${
                                  isHighRisk
                                    ? "border border-red-200 bg-red-50 text-red-700"
                                    : "border border-emerald-200 bg-emerald-50 text-emerald-700"
                                }`}
                              >
                                {isHighRisk
                                  ? "HIGH MEMORY RISK DETECTED"
                                  : "NO HIGH MEMORY RISK DETECTED"}
                              </span>
                            </div>

                            <div className="mt-5 grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
                              <div className="rounded-xl border border-slate-100 bg-white p-4">
                                <p className="text-xs font-semibold uppercase tracking-wider text-slate-400">
                                  Predicted risk probability
                                </p>
                                <p className="mt-2 text-3xl font-bold text-slate-800">
                                  {riskProbability === null
                                    ? "Not available"
                                    : `${Math.max(0, Math.min(100, riskProbability)).toFixed(2)}%`}
                                </p>
                                {riskProbability !== null && (
                                  <div className="mt-3 h-2 overflow-hidden rounded-full bg-slate-100">
                                    <div
                                      className={`h-full rounded-full ${
                                        isHighRisk ? "bg-red-500" : "bg-emerald-500"
                                      }`}
                                      style={{
                                        width: `${Math.max(0, Math.min(100, riskProbability))}%`,
                                      }}
                                    />
                                  </div>
                                )}
                              </div>

                              <div className="rounded-xl border border-slate-100 bg-white p-4">
                                <p className="text-xs font-semibold uppercase tracking-wider text-slate-400">
                                  Classification threshold
                                </p>
                                <p className="mt-2 text-3xl font-bold text-slate-800">
                                  {threshold === null
                                    ? "Not available"
                                    : `${(threshold * 100).toFixed(2)}%`}
                                </p>
                                <p className="mt-2 text-xs text-slate-400">
                                  Model decision threshold
                                </p>
                              </div>

                              <div className="rounded-xl border border-slate-100 bg-white p-4">
                                <p className="text-xs font-semibold uppercase tracking-wider text-slate-400">
                                  Prediction horizon
                                </p>
                                <p className="mt-2 text-3xl font-bold text-slate-800">
                                  {memoryRiskAnalysis.prediction_horizon_minutes ?? 30} min
                                </p>
                                <p className="mt-2 text-xs text-slate-400">
                                  Future memory-risk prediction
                                </p>
                              </div>
                            </div>

                            <div className="mt-4 grid grid-cols-1 gap-3 sm:grid-cols-2">
                              <div className="rounded-xl bg-white/80 p-3">
                                <p className="text-[10px] font-semibold uppercase tracking-wider text-slate-400">
                                  Input history
                                </p>
                                <p className="mt-1 text-sm font-bold text-slate-700">
                                  {memoryRiskAnalysis.input_history_minutes ?? 60} minutes
                                </p>
                              </div>
                              <div className="rounded-xl bg-white/80 p-3">
                                <p className="text-[10px] font-semibold uppercase tracking-wider text-slate-400">
                                  Engineered features
                                </p>
                                <p className="mt-1 text-sm font-bold text-slate-700">
                                  {memoryRiskAnalysis.feature_count ?? "—"} features
                                </p>
                              </div>
                            </div>

                            <div className="mt-4 rounded-xl border border-amber-100 bg-white/80 p-4">
                              <p className="text-xs font-semibold uppercase tracking-wider text-slate-500">
                                Prediction target
                              </p>
                              <p className="mt-2 text-sm leading-relaxed text-slate-700">
                                {memoryRiskAnalysis.target_definition ||
                                  "Predicts whether future peak memory utilization will reach or exceed 85%."}
                              </p>
                            </div>

                            <p className="mt-4 text-[11px] leading-relaxed text-slate-500">
                              This experimental ML prediction uses the selected Alibaba Cluster Trace workload sample.
                              It is an advisory risk signal, not live telemetry from this application's cloud provider.
                              A low predicted risk does not guarantee that memory spikes will not occur. No automatic
                              resource changes are performed.
                            </p>
                          </>
                        );
                      })()}
                    </div>
                  )}

                  {Object.keys(resourceStatistics).length > 0 && (
                    <div className="mt-6 rounded-2xl border border-blue-100 bg-blue-50/50 p-5">
                      <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
                        <div>
                          <h3 className="font-bold text-slate-800">
                            Historical Workload Analysis
                          </h3>
                          <p className="mt-1 text-xs text-slate-500">
                            {workloadSource || "Sample workload dataset"}
                            {workloadMachineId !== undefined
                              ? ` • Machine ${workloadMachineId}`
                              : ""}
                            {forecastMethod ? ` • Forecast: ${forecastMethod}` : ""}
                          </p>
                        </div>
                        <span className="rounded-full bg-white px-3 py-1.5 text-xs font-semibold text-blue-700">
                          {workloadAnalysis.input_metadata?.observations ?? 12} observations
                        </span>
                      </div>

                      <div className="mt-4 grid grid-cols-1 gap-3 md:grid-cols-3">
                        {["cpu", "memory", "disk"].map((resourceName) => {
                          const stats = resourceStatistics[resourceName] ?? {};
                          const trend = workloadTrends[resourceName];
                          return (
                            <div
                              key={resourceName}
                              className="rounded-xl border border-slate-100 bg-white p-4"
                            >
                              <p className="text-xs font-bold uppercase tracking-wider text-slate-400">
                                {resourceName} utilization
                              </p>
                              <p className="mt-2 text-2xl font-bold text-slate-800">
                                {formatPercentValue(stats.current)}
                              </p>
                              <div className="mt-3 space-y-1.5 text-xs text-slate-500">
                                <div className="flex justify-between gap-2">
                                  <span>Average</span>
                                  <strong className="text-slate-700">
                                    {formatPercentValue(stats.average)}
                                  </strong>
                                </div>
                                <div className="flex justify-between gap-2">
                                  <span>Peak</span>
                                  <strong className="text-slate-700">
                                    {formatPercentValue(stats.peak)}
                                  </strong>
                                </div>
                                {trend?.direction && (
                                  <div className="flex justify-between gap-2">
                                    <span>Trend</span>
                                    <strong className={
                                      trend.direction === "INCREASING"
                                        ? "text-amber-600"
                                        : trend.direction === "DECREASING"
                                          ? "text-emerald-600"
                                          : "text-blue-600"
                                    }>
                                      {trend.direction}
                                    </strong>
                                  </div>
                                )}
                              </div>
                            </div>
                          );
                        })}
                      </div>

                      {Object.keys(workloadForecasts).length > 0 && (
                        <div className="mt-5">
                          <h4 className="text-sm font-bold text-slate-800">
                            CPU & Memory Forecast
                          </h4>
                          <div className="mt-3 grid grid-cols-1 gap-3 sm:grid-cols-2">
                            {["15min", "30min"].map((horizon) => {
                              const forecast = workloadForecasts[horizon] ?? {};
                              return (
                                <div
                                  key={horizon}
                                  className="rounded-xl border border-slate-100 bg-white p-4"
                                >
                                  <p className="text-xs font-bold text-slate-500">
                                    {horizon} forecast
                                  </p>
                                  <div className="mt-2 grid grid-cols-2 gap-3">
                                    <div>
                                      <p className="text-[10px] uppercase text-slate-400">CPU</p>
                                      <p className="mt-1 text-lg font-bold text-slate-800">
                                        {formatPercentValue(forecast.cpu_utilization)}
                                      </p>
                                    </div>
                                    <div>
                                      <p className="text-[10px] uppercase text-slate-400">Memory</p>
                                      <p className="mt-1 text-lg font-bold text-slate-800">
                                        {formatPercentValue(forecast.memory_utilization)}
                                      </p>
                                    </div>
                                  </div>
                                </div>
                              );
                            })}
                          </div>
                        </div>
                      )}

                      <p className="mt-4 text-[11px] leading-relaxed text-slate-500">
                        Demonstration workload from Alibaba Cluster Trace 2017.
                        This sample machine is not connected to the selected application's
                        live cloud resources.
                      </p>
                    </div>
                  )}

                  {utilizationEntries.length > 0 && (
                    <div className="mt-6">
                      <h3 className="font-bold text-slate-800">
                        Resource Utilization
                      </h3>
                      <div className="mt-3 grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-4">
                        {utilizationEntries.map(([name, rawValue]) => {
                          const value = clampPercent(rawValue);
                          return (
                            <div
                              key={name}
                              className="rounded-xl border border-slate-100 bg-white p-4"
                            >
                              <div className="flex items-center justify-between gap-2">
                                <span className="text-xs font-semibold capitalize text-slate-500">
                                  {name.replaceAll("_", " ")}
                                </span>
                                <span className="text-sm font-bold text-slate-800">
                                  {value === null ? String(rawValue) : `${value.toFixed(1)}%`}
                                </span>
                              </div>
                              {value !== null && (
                                <div className="mt-3 h-2 overflow-hidden rounded-full bg-slate-100">
                                  <div
                                    className="h-full rounded-full bg-gradient-to-r from-blue-500 to-violet-500"
                                    style={{ width: `${value}%` }}
                                  />
                                </div>
                              )}
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  )}

                  {Array.isArray(reasons) && reasons.length > 0 && (
                    <div className="mt-6">
                      <h3 className="font-bold text-slate-800">
                        Analysis Reasons
                      </h3>
                      <ul className="mt-3 grid grid-cols-1 gap-2 md:grid-cols-2">
                        {reasons.map((reason, index) => (
                          <li
                            key={`${index}-${String(reason)}`}
                            className="flex items-start gap-2 rounded-xl bg-slate-50 p-3 text-sm text-slate-600"
                          >
                            <CheckCircle2 size={16} className="mt-0.5 shrink-0 text-emerald-500" />
                            {typeof reason === "string" ? reason : JSON.stringify(reason)}
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}

                  <details className="mt-6 rounded-xl border border-slate-200 bg-slate-50 p-4">
                    <summary className="cursor-pointer text-sm font-semibold text-slate-700">
                      View complete API response
                    </summary>
                    <pre className="mt-3 max-h-96 overflow-auto whitespace-pre-wrap break-words text-xs text-slate-600">
                      {JSON.stringify(result, null, 2)}
                    </pre>
                  </details>
                </div>
              );
            })()}

            {optimizationHistory.length > 0 && (
              <div className="mt-7 border-t border-slate-100 pt-6">
                <h3 className="font-bold text-slate-800">
                  Previous Optimization Runs
                </h3>
                <div className="mt-3 space-y-3">
                  {optimizationHistory.slice(0, 5).map((item, index) => {
                    const itemRecommendation = item.recommendation ?? item.result?.recommendation ?? {};
                    const itemAction = readOptimizationValue(
                      itemRecommendation.action,
                      itemRecommendation.decision,
                      item.action,
                      item.decision
                    );
                    const itemTitle = readOptimizationValue(
                      itemRecommendation.title,
                      item.title,
                      item.message
                    );
                    const createdAt = readOptimizationValue(
                      item.created_at,
                      item.createdAt,
                      item.timestamp
                    );
                    const itemResourceAnalysis =
                      item.resource_analysis ??
                      item.resourceAnalysis ??
                      item.result?.resource_analysis ??
                      item.result?.resourceAnalysis ??
                      {};
                    const itemMlPrediction =
                      itemResourceAnalysis.ml_analysis?.prediction ??
                      itemResourceAnalysis.mlAnalysis?.prediction ??
                      null;
                    const itemMlConfidence = toFiniteNumber(
                      itemMlPrediction?.confidence,
                      null
                    );
                    const itemMlConfidencePercent =
                      itemMlConfidence === null
                        ? null
                        : itemMlConfidence <= 1
                          ? itemMlConfidence * 100
                          : itemMlConfidence;
                    const itemMemoryRisk =
                      itemResourceAnalysis.memory_risk_analysis ??
                      itemResourceAnalysis.memoryRiskAnalysis ??
                      item.memory_risk_analysis ??
                      item.memoryRiskAnalysis ??
                      null;
                    const itemRiskProbabilityRaw = toFiniteNumber(
                      itemMemoryRisk?.risk_probability_percent,
                      null
                    );
                    const itemRiskProbabilityScore = toFiniteNumber(
                      itemMemoryRisk?.risk_probability,
                      null
                    );
                    const itemRiskProbability =
                      itemRiskProbabilityRaw !== null
                        ? itemRiskProbabilityRaw
                        : itemRiskProbabilityScore !== null
                          ? itemRiskProbabilityScore * 100
                          : null;
                    const hasItemMemoryRisk =
                      itemMemoryRisk?.status === "success" &&
                      typeof itemMemoryRisk?.high_memory_risk === "boolean";
                    return (
                      <div
                        key={item.id ?? item._id ?? `${index}-${createdAt ?? "history"}`}
                        className="flex flex-col gap-2 rounded-xl border border-slate-100 p-4 sm:flex-row sm:items-center sm:justify-between"
                      >
                        <div>
                          <p className="text-sm font-bold text-slate-700">
                            {itemAction ? String(itemAction).toUpperCase() : "Optimization run"}
                          </p>
                          <p className="mt-1 text-xs text-slate-500">
                            {itemTitle || item.application_name || item.applicationName || "Saved recommendation"}
                          </p>
                          {itemMlPrediction?.predicted_condition && (
                            <p className="mt-1 text-[11px] font-medium text-violet-600">
                              ML: {itemMlPrediction.predicted_condition}
                              {itemMlConfidencePercent !== null
                                ? ` • ${itemMlConfidencePercent.toFixed(2)}% confidence`
                                : ""}
                            </p>
                          )}
                          {hasItemMemoryRisk && (
                            <p
                              className={`mt-1 text-[11px] font-semibold ${
                                itemMemoryRisk.high_memory_risk
                                  ? "text-red-600"
                                  : "text-emerald-600"
                              }`}
                            >
                              Memory risk: {itemMemoryRisk.high_memory_risk ? "HIGH" : "NO HIGH RISK"}
                              {itemRiskProbability !== null
                                ? ` • ${Math.max(0, Math.min(100, itemRiskProbability)).toFixed(2)}% probability`
                                : ""}
                              {itemMemoryRisk.prediction_horizon_minutes
                                ? ` • ${itemMemoryRisk.prediction_horizon_minutes} min forecast`
                                : ""}
                            </p>
                          )}
                        </div>
                        <p className="text-xs text-slate-400">
                          {createdAt
                            ? new Date(createdAt).toLocaleString()
                            : "Date not available"}
                        </p>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}
          </section>
        )}

        {/* =================================================== */}
        {/* COST SCENARIO ENGINE */}
        {/* =================================================== */}

        {activeApplication && (
          <CostScenarioPanel application={activeApplication} latestOptimization={optimizationResult} />
        )}

        {/* =================================================== */}
        {/* TOP OPPORTUNITY */}
        {/* =================================================== */}

        {topOpportunity && <div className="mt-10 grid grid-cols-1 gap-6 xl:grid-cols-3">
          <div className="relative overflow-hidden rounded-3xl bg-gradient-to-br from-violet-600 via-blue-600 to-blue-700 p-7 text-white shadow-lg xl:col-span-2">
            <div className="absolute -right-12 -top-12 h-56 w-56 rounded-full bg-white/10 blur-2xl" />

            <div className="relative">
              <div className="flex flex-col gap-6 lg:flex-row lg:items-center lg:justify-between">
                <div>
                  <div className="flex items-center gap-2">
                    <Sparkles size={19} />

                    <p className="text-xs font-bold uppercase tracking-[0.2em] text-blue-100">
                      Top AI Opportunity
                    </p>
                  </div>

                  <h2 className="mt-4 text-2xl font-bold">
                    {topOpportunity.name}
                  </h2>

                  <p className="mt-2 max-w-xl text-sm leading-relaxed text-blue-100">
                    {topOpportunity.recommendation}
                  </p>

                  <div className="mt-6 flex flex-wrap gap-3">
                    <div className="rounded-xl bg-white/15 px-4 py-3 backdrop-blur">
                      <p className="text-[10px] font-bold uppercase tracking-wider text-blue-100">
                        Potential Savings
                      </p>

                      <p className="mt-1 text-xl font-bold">
                        $
                        {
                          topOpportunity.potentialSavings
                        }
                        /month
                      </p>
                    </div>

                    <div className="rounded-xl bg-white/15 px-4 py-3 backdrop-blur">
                      <p className="text-[10px] font-bold uppercase tracking-wider text-blue-100">
                        Utilization Fit Score
                      </p>

                      <p className="mt-1 text-xl font-bold">
                        {formatScore(topOpportunity.optimizationScore)}
                      </p>
                    </div>
                  </div>
                </div>

                <button
                  type="button"
                  onClick={() => {
                    setSelectedApplication(
                      topOpportunity
                    );
                    setSelectedDecision(
                      getDefaultDecision(
                        topOpportunity
                      )
                    );

                    window.scrollTo({
                      top: 850,
                      behavior: "smooth",
                    });
                  }}
                  className="flex shrink-0 items-center justify-center gap-2 rounded-xl bg-white px-5 py-3.5 text-sm font-bold text-blue-700 shadow-lg transition hover:scale-105 hover:bg-blue-50"
                >
                  Analyze Now

                  <ArrowUpRight size={18} />
                </button>
              </div>
            </div>
          </div>

          <div className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm">
            <div className="flex items-center gap-2">
              <Lightbulb
                size={21}
                className="text-violet-600"
              />

              <h3 className="font-bold text-slate-800">
                AI Insight
              </h3>
            </div>

            <p className="mt-4 text-sm leading-relaxed text-slate-500">
              CloudMorph identified{" "}
              <span className="font-bold text-slate-800">
                {highPriorityApps} application
                {highPriorityApps !== 1
                  ? "s"
                  : ""}
              </span>{" "}
              requiring additional optimization review.
            </p>

            <div className="mt-6 rounded-xl bg-emerald-50 p-4">
              <div className="flex items-center gap-3">
                <CircleDollarSign
                  size={22}
                  className="text-emerald-600"
                />

                <div>
                  <p className="text-xs font-medium text-emerald-600">
                    Estimated Annual Savings
                  </p>

                  <p className="mt-1 text-xl font-bold text-emerald-700">
                    $
                    {(
                      totalPotentialSavings *
                      12
                    ).toFixed(2)}
                  </p>
                </div>
              </div>
            </div>

            <button
              type="button"
              onClick={() => {
                setSelectedApplication(
                  topOpportunity
                );
                setSelectedDecision(
                  getDefaultDecision(
                    topOpportunity
                  )
                );

                window.scrollTo({
                  top: 850,
                  behavior: "smooth",
                });
              }}
              className="mt-5 flex w-full items-center justify-center gap-2 rounded-xl border border-violet-100 bg-violet-50 py-3 text-sm font-bold text-violet-600 transition hover:bg-violet-100"
            >
              <Sparkles size={17} />
              Review AI Opportunity
            </button>
          </div>
        </div>}

        {/* =================================================== */}
        {/* APPLICATION OPTIMIZATION */}
        {/* =================================================== */}

        <div className="mt-10">
          <div className="flex flex-col gap-5 xl:flex-row xl:items-end xl:justify-between">
            <div>
              <div className="flex items-center gap-2">
                <BrainCircuit
                  size={23}
                  className="text-violet-600"
                />

                <h2 className="text-xl font-bold text-slate-800">
                  Application Optimization
                </h2>
              </div>

              <p className="mt-2 text-sm text-slate-500">
                Select an application to view detailed AI-powered
                optimization recommendations.
              </p>
            </div>

            {/* FILTERS */}

            <div className="flex flex-col gap-3 md:flex-row">
              <div className="relative">
                <Search
                  size={18}
                  className="absolute left-4 top-1/2 -translate-y-1/2 text-slate-400"
                />

                {searchQuery && (
                  <button
                    type="button"
                    onClick={() =>
                      setSearchQuery("")
                    }
                    className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-400 transition hover:text-slate-700"
                  >
                    <X size={16} />
                  </button>
                )}

                <input
                  type="text"
                  value={searchQuery}
                  onChange={(event) =>
                    setSearchQuery(
                      event.target.value
                    )
                  }
                  placeholder="Search applications..."
                  className="w-full rounded-xl border border-slate-200 bg-white py-3 pl-11 pr-10 text-sm outline-none transition focus:border-blue-400 focus:ring-4 focus:ring-blue-100 md:w-64"
                />
              </div>

              <div className="relative">
                <Filter
                  size={17}
                  className="pointer-events-none absolute left-4 top-1/2 -translate-y-1/2 text-slate-400"
                />

                <select
                  value={providerFilter}
                  onChange={(event) =>
                    setProviderFilter(
                      event.target.value
                    )
                  }
                  className="appearance-none rounded-xl border border-slate-200 bg-white py-3 pl-11 pr-10 text-sm font-medium text-slate-600 outline-none transition focus:border-blue-400 focus:ring-4 focus:ring-blue-100"
                >
                  <option value="All">
                    All Providers
                  </option>

                  <option value="AWS">
                    AWS
                  </option>

                  <option value="GCP">
                    GCP
                  </option>
                </select>

                <ChevronDown
                  size={16}
                  className="pointer-events-none absolute right-4 top-1/2 -translate-y-1/2 text-slate-400"
                />
              </div>

              <div className="relative">
                <SlidersHorizontal
                  size={17}
                  className="pointer-events-none absolute left-4 top-1/2 -translate-y-1/2 text-slate-400"
                />

                <select
                  value={sortBy}
                  onChange={(event) =>
                    setSortBy(
                      event.target.value
                    )
                  }
                  className="appearance-none rounded-xl border border-slate-200 bg-white py-3 pl-11 pr-10 text-sm font-medium text-slate-600 outline-none transition focus:border-blue-400 focus:ring-4 focus:ring-blue-100"
                >
                  <option value="score">
                    Optimization Score
                  </option>

                  <option value="savings">
                    Potential Savings
                  </option>

                  <option value="cost">
                    Monthly Cost
                  </option>

                  <option value="health">
                    Health Score
                  </option>
                </select>

                <ChevronDown
                  size={16}
                  className="pointer-events-none absolute right-4 top-1/2 -translate-y-1/2 text-slate-400"
                />
              </div>
            </div>
          </div>

          <div className="mt-6 flex items-center justify-between">
            <p className="text-sm text-slate-500">
              Showing{" "}
              <span className="font-bold text-slate-700">
                {filteredApplications.length}
              </span>{" "}
              of{" "}
              <span className="font-bold text-slate-700">
                {applications.length}
              </span>{" "}
              applications
            </p>

            <div className="hidden items-center gap-2 text-xs font-medium text-slate-400 md:flex">
              <Activity size={15} />
              AI monitoring active
            </div>
          </div>

          {/* APPLICATION CARDS */}

          {filteredApplications.length > 0 ? (
            <div className="mt-6 grid grid-cols-1 gap-6 lg:grid-cols-2">
              {filteredApplications.map(
                (application) => {
                  const AppIcon =
                    application.icon;

                  const scoreStyle =
                    getScoreColor(
                      application.optimizationScore
                    );

                  return (
                    <div
                      key={application.id}
                      className="group relative overflow-hidden rounded-3xl border border-slate-200 bg-white p-6 shadow-sm transition duration-300 hover:-translate-y-1 hover:border-blue-200 hover:shadow-xl"
                    >
                      <div className="absolute right-0 top-0 h-32 w-32 rounded-full bg-blue-100/40 blur-3xl opacity-0 transition group-hover:opacity-100" />

                      <div className="relative">
                        {/* HEADER */}

                        <div className="flex items-start justify-between gap-4">
                          <div className="flex items-start gap-4">
                            <div className="flex h-14 w-14 shrink-0 items-center justify-center rounded-2xl bg-gradient-to-br from-blue-600 to-violet-600 text-white shadow-lg shadow-blue-200">
                              <AppIcon size={26} />
                            </div>

                            <div>
                              <div className="flex flex-wrap items-center gap-2">
                                <h3 className="text-lg font-bold text-slate-800">
                                  {application.name}
                                </h3>

                                <span
                                  className={`rounded-full border px-2.5 py-1 text-[10px] font-bold ${getStatusStyle(
                                    application.status
                                  )}`}
                                >
                                  {application.status}
                                </span>
                              </div>

                              <p className="mt-1 text-sm text-slate-500">
                                {
                                  application.description
                                }
                              </p>

                              <div className="mt-3 flex flex-wrap gap-2">
                                <span className="rounded-lg bg-slate-100 px-3 py-1.5 text-[10px] font-bold text-slate-600">
                                  {
                                    application.provider
                                  }
                                </span>

                                <span className="rounded-lg bg-blue-50 px-3 py-1.5 text-[10px] font-bold text-blue-600">
                                  {
                                    application.category
                                  }
                                </span>
                              </div>
                            </div>
                          </div>

                          <div
                            className={`flex h-16 w-16 shrink-0 flex-col items-center justify-center rounded-2xl border ${scoreStyle.bg} ${scoreStyle.border}`}
                          >
                            <span
                              className={`text-lg font-bold ${scoreStyle.text}`}
                            >
                              {formatScore(application.optimizationScore)}
                            </span>

                            <span className="text-[8px] font-bold uppercase text-slate-400">
                              Score
                            </span>
                          </div>
                        </div>

                        {/* SCORE */}

                        <div className="mt-7">
                          <div className="flex items-center justify-between">
                            <p className="text-xs font-semibold text-slate-500">
                              Utilization Fit Score
                            </p>

                            <p
                              className={`text-xs font-bold ${scoreStyle.text}`}
                            >
                              {Number.isFinite(application.optimizationScore)
                                ? application.optimizationScore >= 85
                                  ? "Excellent"
                                  : application.optimizationScore >= 70
                                    ? "Good"
                                    : "Needs Attention"
                                : "Pending analysis"}
                            </p>
                          </div>

                          <div className="mt-3 h-2.5 overflow-hidden rounded-full bg-slate-100">
                            <div
                              className={`h-full rounded-full transition-all duration-700 ${scoreStyle.progress}`}
                              style={{
                                width: `${Number.isFinite(application.optimizationScore) ? Math.max(0, Math.min(100, application.optimizationScore)) : 0}%`,
                              }}
                            />
                          </div>
                        </div>

                        {/* RESOURCE GRID */}

                        <div className="mt-6 grid grid-cols-2 gap-3 sm:grid-cols-4">
                          <ResourceMini
                            icon={Cpu}
                            label="Compute"
                            score={
                              application
                                .resources
                                .compute
                            }
                            progressColor={getResourceColor(
                              application
                                .resources
                                .compute
                            )}
                          />

                          <ResourceMini
                            icon={Database}
                            label="Database"
                            score={
                              application
                                .resources
                                .database
                            }
                            progressColor={getResourceColor(
                              application
                                .resources
                                .database
                            )}
                          />

                          <ResourceMini
                            icon={HardDrive}
                            label="Storage"
                            score={
                              application
                                .resources
                                .storage
                            }
                            progressColor={getResourceColor(
                              application
                                .resources
                                .storage
                            )}
                          />

                          <ResourceMini
                            icon={Network}
                            label="Network"
                            score={
                              application
                                .resources
                                .network
                            }
                            progressColor={getResourceColor(
                              application
                                .resources
                                .network
                            )}
                          />
                        </div>

                        {/* METRICS */}

                        <div className="mt-6 grid grid-cols-2 gap-3 md:grid-cols-4">
                          <MiniMetric
                            icon={DollarSign}
                            label="Monthly Cost"
                            value={`$${application.monthlyCost.toFixed(
                              2
                            )}`}
                            color="text-blue-600"
                          />

                          <MiniMetric
                            icon={TrendingDown}
                            label="Savings"
                            value={`$${application.potentialSavings.toFixed(
                              2
                            )}`}
                            color="text-emerald-600"
                          />

                          <MiniMetric
                            icon={ShieldCheck}
                            label="Health"
                            value={formatScore(application.healthScore)}
                            color="text-violet-600"
                          />

                          <MiniMetric
                            icon={Cloud}
                            label="Services"
                            value={
                              application.services
                            }
                            color="text-orange-600"
                          />
                        </div>

                        {/* RECOMMENDATION */}

                        <div className="mt-6 rounded-2xl bg-gradient-to-r from-violet-50 to-blue-50 p-4">
                          <div className="flex gap-3">
                            <div className="mt-0.5 text-violet-600">
                              <Sparkles size={19} />
                            </div>

                            <div>
                              <p className="text-xs font-bold text-slate-700">
                                AI Recommendation
                              </p>

                              <p className="mt-1 text-xs leading-relaxed text-slate-500">
                                {
                                  application.recommendation
                                }
                              </p>
                            </div>
                          </div>
                        </div>

                        {/* FOOTER */}

                        <div className="mt-6 flex flex-col gap-4 border-t border-slate-100 pt-5 sm:flex-row sm:items-center sm:justify-between">
                          <div className="flex items-center gap-2">
                            <Activity
                              size={16}
                              className={getUsageStyle(
                                application.usage
                              )}
                            />

                            <span className="text-xs text-slate-400">
                              Usage:
                            </span>

                            <span
                              className={`text-xs font-bold ${getUsageStyle(
                                application.usage
                              )}`}
                            >
                              {
                                application.usage
                              }
                            </span>
                          </div>

                          <div className="flex gap-2">
                            <button
                              type="button"
                              onClick={() =>
                                setQuickViewApplication(
                                  application
                                )
                              }
                              className="rounded-xl border border-slate-200 px-4 py-3 text-xs font-bold text-slate-600 transition hover:border-blue-200 hover:bg-blue-50 hover:text-blue-600"
                            >
                              Quick View
                            </button>

                            <button
                              type="button"
                              onClick={() =>
                                handleAnalyze(
                                  application
                                )
                              }
                              className="flex items-center justify-center gap-2 rounded-xl bg-slate-900 px-5 py-3 text-sm font-bold text-white transition hover:bg-blue-600"
                            >
                              <BrainCircuit
                                size={17}
                              />

                              Analyze AI

                              <ArrowUpRight
                                size={16}
                              />
                            </button>
                          </div>
                        </div>
                      </div>
                    </div>
                  );
                }
              )}
            </div>
          ) : (
            <div className="mt-6 flex flex-col items-center justify-center rounded-3xl border border-dashed border-slate-300 bg-white p-16 text-center">
              <div className="flex h-16 w-16 items-center justify-center rounded-2xl bg-slate-100 text-slate-400">
                <Search size={28} />
              </div>

              <h3 className="mt-5 text-lg font-bold text-slate-700">
                No Applications Found
              </h3>

              <p className="mt-2 max-w-sm text-sm text-slate-500">
                Try changing your search query or provider
                filter.
              </p>

              <button
                type="button"
                onClick={resetFilters}
                className="mt-6 rounded-xl bg-slate-900 px-5 py-3 text-sm font-bold text-white transition hover:bg-blue-600"
              >
                Reset Filters
              </button>
            </div>
          )}
        </div>

        {/* =================================================== */}
        {/* BOTTOM SUMMARY */}
        {/* =================================================== */}

        <div className="mt-10 overflow-hidden rounded-3xl border border-blue-100 bg-gradient-to-r from-blue-50 via-white to-violet-50 p-8">
          <div className="flex flex-col gap-7 lg:flex-row lg:items-center lg:justify-between">
            <div className="flex items-start gap-5">
              <div className="rounded-2xl bg-gradient-to-br from-blue-600 to-violet-600 p-4 text-white shadow-lg">
                <BrainCircuit size={28} />
              </div>

              <div>
                <p className="text-xs font-bold uppercase tracking-[0.2em] text-violet-600">
                  AI Optimization Summary
                </p>

                <h2 className="mt-3 text-2xl font-bold text-slate-800">
                  Your infrastructure has optimization potential.
                </h2>

                <p className="mt-3 max-w-2xl text-sm leading-relaxed text-slate-500">
                  CloudMorph AI identified optimization
                  opportunities across{" "}
                  <span className="font-bold text-slate-700">
                    {applications.length} applications
                  </span>
                  . Potential savings are estimated at{" "}
                  <span className="font-bold text-emerald-600">
                    ${totalPotentialSavings.toFixed(2)}
                  </span>{" "}
                  per month.
                </p>
              </div>
            </div>

            <div className="flex shrink-0 items-center gap-4 rounded-2xl bg-white p-5 shadow-sm">
              <div className="rounded-xl bg-emerald-50 p-3 text-emerald-600">
                <TrendingDown size={25} />
              </div>

              <div>
                <p className="text-xs text-slate-400">
                  Potential Savings
                </p>

                <p className="mt-1 text-2xl font-bold text-slate-800">
                  ${totalPotentialSavings.toFixed(2)}

                  <span className="ml-1 text-sm text-slate-400">
                    /month
                  </span>
                </p>
              </div>
            </div>
          </div>
        </div>

        <div className="h-6" />
      </main>

      {/* ===================================================== */}
      {/* QUICK VIEW MODAL */}
      {/* ===================================================== */}

      {quickViewApplication && (
        <QuickViewModal
          application={quickViewApplication}
          onClose={() => setQuickViewApplication(null)}
          onAnalyze={() => handleAnalyze(quickViewApplication)}
          onViewDetails={() => handleViewDetails(quickViewApplication)}
        />
      )}
    </div>
  );
}

/* ========================================================= */
/* STATUS CARD */
/* ========================================================= */

function StatusCard({
  icon: Icon,
  iconStyle,
  label,
  value,
}) {
  return (
    <div className="rounded-2xl border border-white/10 bg-white/5 p-5 backdrop-blur">
      <div className="flex items-center gap-3">
        <div
          className={`rounded-xl p-3 ${iconStyle}`}
        >
          <Icon size={22} />
        </div>

        <div>
          <p className="text-xs text-slate-400">
            {label}
          </p>

          <p className="mt-1 text-sm font-bold text-white">
            {value}
          </p>
        </div>
      </div>
    </div>
  );
}

/* ========================================================= */
/* QUICK INSIGHT */
/* ========================================================= */

function QuickInsight({
  icon: Icon,
  iconColor,
  title,
  value,
  description,
}) {
  return (
    <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
      <div className="flex items-center justify-between">
        <div
          className={`rounded-xl bg-slate-50 p-3 ${iconColor}`}
        >
          <Icon size={21} />
        </div>

        <ArrowUpRight
          size={17}
          className="text-slate-300"
        />
      </div>

      <p className="mt-5 text-xs font-medium text-slate-400">
        {title}
      </p>

      <p className="mt-2 text-2xl font-bold text-slate-800">
        {value}
      </p>

      <p className="mt-1 text-xs text-slate-400">
        {description}
      </p>
    </div>
  );
}

/* ========================================================= */
/* WORKFLOW STEP */
/* ========================================================= */

function WorkflowStep({
  number,
  title,
  description,
  icon: Icon,
}) {
  return (
    <div className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm">
      <div className="flex items-center gap-3">
        <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-blue-50 text-blue-600">
          <Icon size={19} />
        </div>

        <div>
          <p className="text-[9px] font-bold uppercase tracking-wider text-blue-600">
            {number}
          </p>

          <p className="text-sm font-bold text-slate-700">
            {title}
          </p>

          <p className="text-[10px] text-slate-400">
            {description}
          </p>
        </div>
      </div>
    </div>
  );
}

/* ========================================================= */
/* DECISION MINI STAT */
/* ========================================================= */

function DecisionMiniStat({
  label,
  value,
}) {
  return (
    <div className="rounded-xl bg-white p-3 shadow-sm">
      <p className="text-[9px] font-medium text-slate-400">
        {label}
      </p>

      <p className="mt-1 text-sm font-bold text-slate-800">
        {value}
      </p>
    </div>
  );
}

/* ========================================================= */
/* INFRASTRUCTURE BOX */
/* ========================================================= */

function InfrastructureBox({
  label,
  value,
  icon: Icon,
}) {
  return (
    <div className="rounded-2xl border border-slate-200 bg-white p-4">
      <div className="flex items-center gap-3">
        <div className="rounded-xl bg-slate-100 p-2.5 text-slate-600">
          <Icon size={18} />
        </div>

        <div className="min-w-0">
          <p className="text-[10px] font-medium text-slate-400">
            {label}
          </p>

          <p className="mt-1 truncate text-xs font-bold text-slate-700">
            {value}
          </p>
        </div>
      </div>
    </div>
  );
}

/* ========================================================= */
/* IMPACT METRIC */
/* ========================================================= */

function ImpactMetric({
  icon: Icon,
  label,
  value,
  helper,
  positive,
}) {
  return (
    <div className="rounded-xl border border-slate-100 bg-white p-4">
      <div className="flex items-center gap-2">
        <Icon
          size={16}
          className={
            positive
              ? "text-emerald-500"
              : "text-slate-400"
          }
        />

        <p className="text-[10px] font-medium text-slate-400">
          {label}
        </p>
      </div>

      <p className="mt-2 text-sm font-bold text-slate-800">
        {value}
      </p>

      <p
        className={`mt-1 text-[10px] font-medium ${
          positive
            ? "text-emerald-600"
            : "text-slate-400"
        }`}
      >
        {helper}
      </p>
    </div>
  );
}

/* ========================================================= */
/* COST BAR */
/* ========================================================= */

function CostBar({
  label,
  amount,
  maxAmount,
  icon: Icon,
  type,
}) {
  const percentage =
    maxAmount > 0
      ? Math.min(
          (amount / maxAmount) * 100,
          100
        )
      : 0;

  return (
    <div className="rounded-2xl border border-slate-100 bg-slate-50 p-5">
      <div className="flex items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <div
            className={`rounded-xl p-3 ${
              type === "projected"
                ? "bg-emerald-50 text-emerald-600"
                : "bg-blue-50 text-blue-600"
            }`}
          >
            <Icon size={20} />
          </div>

          <div>
            <p className="text-xs text-slate-400">
              {label}
            </p>

            <p className="mt-1 text-xl font-bold text-slate-800">
              ${amount.toFixed(2)}
            </p>
          </div>
        </div>

        <span
          className={`text-xs font-bold ${
            type === "projected"
              ? "text-emerald-600"
              : "text-blue-600"
          }`}
        >
          {percentage.toFixed(0)}%
        </span>
      </div>

      <div className="mt-5 h-3 overflow-hidden rounded-full bg-white">
        <div
          className={`h-full rounded-full transition-all duration-700 ${
            type === "projected"
              ? "bg-emerald-500"
              : "bg-blue-500"
          }`}
          style={{
            width: `${percentage}%`,
          }}
        />
      </div>
    </div>
  );
}

/* ========================================================= */
/* MINI METRIC */
/* ========================================================= */

function MiniMetric({
  icon: Icon,
  label,
  value,
  color,
}) {
  return (
    <div className="rounded-xl bg-slate-50 p-3">
      <div className="flex items-center gap-1.5">
        <Icon
          size={14}
          className={color}
        />

        <p className="truncate text-[9px] font-medium text-slate-400">
          {label}
        </p>
      </div>

      <p className="mt-2 text-sm font-bold text-slate-700">
        {value}
      </p>
    </div>
  );
}

/* ========================================================= */
/* RESOURCE MINI */
/* ========================================================= */

function ResourceMini({
  icon: Icon,
  label,
  score,
  progressColor,
}) {
  return (
    <div className="rounded-xl border border-slate-100 bg-white p-3">
      <div className="flex items-center justify-between">
        <Icon
          size={14}
          className="text-slate-400"
        />

        <span className="text-[10px] font-bold text-slate-600">
          {Number.isFinite(score) ? `${Math.round(score)}%` : "N/A"}
        </span>
      </div>

      <p className="mt-2 text-[10px] font-medium text-slate-500">
        {label}
      </p>

      <div className="mt-2 h-1.5 overflow-hidden rounded-full bg-slate-100">
        <div
          className={`h-full rounded-full ${progressColor}`}
          style={{
            width: `${Number.isFinite(score) ? Math.max(0, Math.min(100, score)) : 0}%`,
          }}
        />
      </div>
    </div>
  );
}

/* ========================================================= */
/* QUICK VIEW MODAL */
/* ========================================================= */

function QuickViewModal({
  application,
  onClose,
  onAnalyze,
  onViewDetails,
}) {
  const AppIcon = application.icon;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/50 p-4 backdrop-blur-sm">
      <div className="w-full max-w-lg overflow-hidden rounded-3xl bg-white shadow-2xl">
        <div className="bg-gradient-to-r from-blue-600 to-violet-600 p-6 text-white">
          <div className="flex items-start justify-between">
            <div className="flex items-center gap-4">
              <div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-white/15 backdrop-blur">
                <AppIcon size={27} />
              </div>

              <div>
                <p className="text-xs font-medium text-blue-100">
                  Application Overview
                </p>

                <h2 className="mt-1 text-xl font-bold">
                  {application.name}
                </h2>
              </div>
            </div>

            <button
              type="button"
              onClick={onClose}
              className="rounded-xl bg-white/10 p-2 text-white transition hover:bg-white/20"
            >
              <X size={20} />
            </button>
          </div>
        </div>

        <div className="p-6">
          <p className="text-sm leading-relaxed text-slate-500">
            {application.description}
          </p>

          <div className="mt-6 grid grid-cols-2 gap-4">
            <ModalMetric
              label="Monthly Cost"
              value={`$${application.monthlyCost.toFixed(
                2
              )}`}
            />

            <ModalMetric
              label="Potential Savings"
              value={`$${application.potentialSavings.toFixed(
                2
              )}`}
            />

            <ModalMetric
              label="Health Score"
              value={formatScore(application.healthScore)}
            />

            <ModalMetric
              label="Utilization Fit Score"
              value={formatScore(application.optimizationScore)}
            />
          </div>

          <div className="mt-6 rounded-2xl bg-violet-50 p-4">
            <div className="flex gap-3">
              <Sparkles
                size={19}
                className="shrink-0 text-violet-600"
              />

              <div>
                <p className="text-xs font-bold text-violet-700">
                  AI Recommendation
                </p>

                <p className="mt-1 text-xs leading-relaxed text-slate-600">
                  {application.recommendation}
                </p>
              </div>
            </div>
          </div>

          <div className="mt-6 flex flex-col gap-3 sm:flex-row">
            <button
              type="button"
              onClick={onViewDetails}
              className="flex flex-1 items-center justify-center gap-2 rounded-xl border border-slate-200 px-4 py-3 text-sm font-bold text-slate-600 transition hover:bg-slate-50"
            >
              View Details

              <ArrowUpRight size={16} />
            </button>

            <button
              type="button"
              onClick={onAnalyze}
              className="flex flex-1 items-center justify-center gap-2 rounded-xl bg-slate-900 px-4 py-3 text-sm font-bold text-white transition hover:bg-blue-600"
            >
              <BrainCircuit size={17} />

              Analyze with AI
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

/* ========================================================= */
/* MODAL METRIC */
/* ========================================================= */

function ModalMetric({
  label,
  value,
}) {
  return (
    <div className="rounded-xl bg-slate-50 p-4">
      <p className="text-xs text-slate-400">
        {label}
      </p>

      <p className="mt-2 text-lg font-bold text-slate-800">
        {value}
      </p>
    </div>
  );
}


/* ========================================================= */
/* COST SCENARIO PANEL */
/* ========================================================= */

function CostScenarioPanel({ application, latestOptimization }) {
  const [scenarioName, setScenarioName] = useState("");
  const [machineType, setMachineType] = useState("");
  const [provider, setProvider] = useState(
    String(application?.provider || "AZURE").toUpperCase()
  );
  const [osType, setOsType] = useState("linux");
  const [providerPricing, setProviderPricing] = useState(null);
  const [isFetchingPricing, setIsFetchingPricing] = useState(false);
  const [vcpu, setVcpu] = useState("");
  const [memoryGb, setMemoryGb] = useState("");
  const [storageGb, setStorageGb] = useState("");
  const [region, setRegion] = useState(application?.region || "");
  const [proposedMonthlyCost, setProposedMonthlyCost] = useState("");
  const [currency, setCurrency] = useState("USD");

  // Cross-cloud VM comparison
  const [compareVcpu, setCompareVcpu] = useState("2");
  const [compareRamGb, setCompareRamGb] = useState("4");
  const [awsRegion, setAwsRegion] = useState("ap-south-1");
  const [azureRegion, setAzureRegion] = useState("centralindia");
  const [gcpRegion, setGcpRegion] = useState("asia-south1");
  const [comparisonResult, setComparisonResult] = useState(null);
  const [isComparingClouds, setIsComparingClouds] = useState(false);
  const [comparisonError, setComparisonError] = useState("");

  const [scenarios, setScenarios] = useState([]);
  const [isLoadingScenarios, setIsLoadingScenarios] = useState(false);
  const [isSavingScenario, setIsSavingScenario] = useState(false);
  const [deletingScenarioId, setDeletingScenarioId] = useState(null);
  const [scenarioError, setScenarioError] = useState("");
  const [scenarioSuccess, setScenarioSuccess] = useState("");

  // Final Recommendation Orchestrator state
  const [finalRecommendation, setFinalRecommendation] = useState(null);
  const [isLoadingFinalRecommendation, setIsLoadingFinalRecommendation] = useState(false);
  const [finalRecommendationError, setFinalRecommendationError] = useState("");

  // Unified view: show the latest resource-engine decision alongside the
  // latest saved cost scenario safety result. Neither engine overrides the other.
  const latestResourceRecommendation =
    latestOptimization?.recommendation ??
    latestOptimization?.result?.recommendation ??
    {};
  const latestResourceAction = String(
    latestResourceRecommendation.action ??
    latestResourceRecommendation.decision ??
    latestOptimization?.action ??
    latestOptimization?.decision ??
    ""
  ).toUpperCase();
  const latestSavedScenario = scenarios[0] ?? null;
  const latestScenarioSafety = latestSavedScenario?.safety_assessment ?? {};

  const formatMoney = (value, code = currency) => {
    const amount = toFiniteNumber(value, null);
    if (amount === null) return "Not available";
    try {
      return new Intl.NumberFormat(undefined, {
        style: "currency",
        currency: code,
        maximumFractionDigits: 2,
      }).format(amount);
    } catch {
      return `${code} ${amount.toFixed(2)}`;
    }
  };

  // Hourly prices are small values; retain precision so close provider prices
  // are not displayed as identical after rounding to two decimal places.
  const formatHourlyMoney = (value, code = currency) => {
    const amount = toFiniteNumber(value, null);
    if (amount === null) return "Not available";
    try {
      return new Intl.NumberFormat(undefined, {
        style: "currency",
        currency: code,
        minimumFractionDigits: 4,
        maximumFractionDigits: 6,
      }).format(amount);
    } catch {
      return `${code} ${amount.toFixed(6)}`;
    }
  };

  const loadScenarios = async () => {
    if (!application?.id) return;
    setIsLoadingScenarios(true);
    setScenarioError("");
    try {
      const response = await API.get(
        `/optimization/applications/${encodeURIComponent(application.id)}/cost-scenarios`
      );
      const payload = response?.data ?? response;
      const items = Array.isArray(payload)
        ? payload
        : payload?.scenarios ?? [];
      setScenarios(Array.isArray(items) ? items : []);
    } catch (error) {
      setScenarioError(
        error.response?.data?.detail ||
          error.message ||
          "Unable to load saved cost scenarios."
      );
    } finally {
      setIsLoadingScenarios(false);
    }
  };

  const loadFinalRecommendation = async () => {
    if (!application?.id) {
      setFinalRecommendation(null);
      return;
    }

    setIsLoadingFinalRecommendation(true);
    setFinalRecommendationError("");

    try {
      const response = await API.get(
        `/optimization/applications/${encodeURIComponent(application.id)}/final-recommendation`
      );
      const payload = response?.data ?? response;

      if (payload?.status !== "success" || !payload?.final_recommendation) {
        throw new Error(
          payload?.detail ||
            payload?.message ||
            "The final recommendation is not available yet."
        );
      }

      setFinalRecommendation(payload);
    } catch (error) {
      setFinalRecommendation(null);
      const status = error.response?.status;
      const detail =
        error.response?.data?.detail ||
        error.response?.data?.message ||
        error.message;

      setFinalRecommendationError(
        status === 409
          ? "Run forecast-aware Resource Analysis first to generate the final recommendation."
          : String(detail || "Unable to load the final recommendation.")
      );
    } finally {
      setIsLoadingFinalRecommendation(false);
    }
  };

  const handleDeleteScenario = async (scenarioId, scenarioName) => {
    if (!application?.id || !scenarioId || deletingScenarioId) return;

    const confirmed = window.confirm(
      `Delete the saved cost scenario "${scenarioName || "Cost scenario"}"? This cannot be undone.`
    );
    if (!confirmed) return;

    setScenarioError("");
    setScenarioSuccess("");
    setDeletingScenarioId(scenarioId);

    try {
      await API.delete(
        `/optimization/applications/${encodeURIComponent(application.id)}/cost-scenarios/${encodeURIComponent(scenarioId)}`
      );
      setScenarios((current) =>
        current.filter((item) => item.scenario_id !== scenarioId)
      );
      setScenarioSuccess("Cost scenario deleted successfully.");
    } catch (error) {
      setScenarioError(
        error.response?.data?.detail ||
          error.message ||
          "Unable to delete this cost scenario."
      );
    } finally {
      setDeletingScenarioId(null);
    }
  };

  useEffect(() => {
    setScenarioName("");
    setMachineType("");
    setProvider(String(application?.provider || "AZURE").toUpperCase());
    setOsType("linux");
    setProviderPricing(null);
    setVcpu("");
    setMemoryGb("");
    setStorageGb("");
    setRegion(application?.region || "");
    setProposedMonthlyCost("");
    setCurrency("USD");
    setScenarioSuccess("");
    setScenarioError("");
    setScenarios([]);
    loadScenarios();
    // The selected application ID is the intended reload trigger.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [application?.id]);

  // Refresh the orchestrator whenever the selected application or its latest
  // forecast-aware resource recommendation changes.
  useEffect(() => {
    setFinalRecommendation(null);
    loadFinalRecommendation();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [
    application?.id,
    latestOptimization?.recommendation_id,
    latestOptimization?.analysis_metadata?.analyzed_at,
  ]);

  const handleFetchProviderPricing = async () => {
    setScenarioError("");
    setScenarioSuccess("");
    setProviderPricing(null);

    const selectedProvider = String(provider || "").trim().toUpperCase();
    const selectedSku = machineType.trim();
    const selectedRegion = region.trim();

    if (!["AZURE", "AWS", "GCP"].includes(selectedProvider)) {
      setScenarioError("Select a supported cloud provider: Azure, AWS, or GCP.");
      return;
    }
    if (!selectedSku || !selectedRegion) {
      setScenarioError("Enter the VM SKU / machine type and region before fetching provider pricing.");
      return;
    }

    setIsFetchingPricing(true);
    try {
      const response = await API.get("/optimization/provider-pricing", {
        params: {
          provider: selectedProvider,
          sku: selectedSku,
          region: selectedRegion,
          os_type: osType,
          currency,
        },
        // The first AWS public price-list download can be large.
        // Allow enough time for the backend to download and parse it.
        timeout: 180000,
      });
      const pricing = response?.data ?? response;
      if (pricing?.status !== "success" || pricing?.pricing_verified !== true) {
        setProviderPricing(pricing);
        setScenarioError(
          pricing?.message ||
            pricing?.detail ||
            `Provider pricing is not available for this configuration (${pricing?.status || "unknown status"}).`
        );
        return;
      }
      setProviderPricing(pricing);
      setScenarioSuccess("Verified public provider pricing retrieved successfully.");
    } catch (error) {
      setScenarioError(
        error.response?.data?.detail ||
          error.response?.data?.message ||
          error.message ||
          "Unable to fetch provider pricing."
      );
    } finally {
      setIsFetchingPricing(false);
    }
  };

  const handleCreateScenario = async (event) => {
    event.preventDefault();
    setScenarioError("");
    setScenarioSuccess("");

    const name = scenarioName.trim();
    const proposedCost = toFiniteNumber(proposedMonthlyCost, null);

    if (!name) {
      setScenarioError("Enter a name for this cost scenario.");
      return;
    }

    if (proposedCost !== null && proposedCost <= 0) {
      setScenarioError("Proposed monthly cost must be greater than zero.");
      return;
    }

    const canUseProviderPricing =
      ["AZURE", "AWS", "GCP"].includes(String(provider).toUpperCase()) &&
      Boolean(machineType.trim()) &&
      Boolean(region.trim());

    if (proposedCost === null && !canUseProviderPricing) {
      setScenarioError(
        "Enter a proposed monthly cost, or provide a cloud provider, VM SKU and region to fetch provider pricing."
      );
      return;
    }

    const parseOptionalResource = (value, label) => {
      if (value === "") return null;
      const parsed = Number(value);
      if (!Number.isFinite(parsed) || parsed <= 0) {
        throw new Error(`${label} must be a number greater than zero.`);
      }
      return parsed;
    };

    let proposedConfiguration;
    try {
      proposedConfiguration = {
        provider: String(provider).toUpperCase(),
        sku: machineType.trim() || null,
        os_type: osType,
        machine_type: machineType.trim() || null,
        vcpu: parseOptionalResource(vcpu, "vCPU"),
        memory_gb: parseOptionalResource(memoryGb, "Memory"),
        storage_gb: parseOptionalResource(storageGb, "Storage"),
        region: region.trim() || null,
      };
    } catch (error) {
      setScenarioError(error.message);
      return;
    }

    setIsSavingScenario(true);
    try {
      const response = await API.post(
        `/optimization/applications/${encodeURIComponent(application.id)}/cost-scenarios`,
        {
          scenario_name: name,
          proposed_configuration: proposedConfiguration,
          ...(proposedCost !== null ? { proposed_monthly_cost: proposedCost } : {}),
          currency,
        }
      );

      const created = response?.data ?? response;
      setScenarioSuccess("Cost scenario evaluated and saved successfully.");
      setScenarioName("");
      setMachineType("");
      setVcpu("");
      setMemoryGb("");
      setStorageGb("");
      setProposedMonthlyCost("");
      setProviderPricing(null);

      await loadScenarios();
      await loadFinalRecommendation();

      // Keep the freshly created response visible if the subsequent history
      // refresh is delayed or returns an empty list.
      if (created?.scenario_id) {
        setScenarios((current) => {
          const exists = current.some(
            (item) => item.scenario_id === created.scenario_id
          );
          return exists ? current : [created, ...current];
        });
      }
    } catch (error) {
      setScenarioError(
        error.response?.data?.detail ||
          error.message ||
          "Unable to create the cost scenario."
      );
    } finally {
      setIsSavingScenario(false);
    }
  };

  const handleCompareClouds = async (event) => {
    event.preventDefault();
    setIsComparingClouds(true);
    setComparisonError("");
    setComparisonResult(null);

    try {
      const response = await API.get("/optimization/compare-vm-costs", {
        params: {
          required_vcpu: Number(compareVcpu),
          required_ram_gb: Number(compareRamGb),
          aws_region: awsRegion.trim(),
          azure_region: azureRegion.trim(),
          gcp_region: gcpRegion.trim(),
          currency,
          os_type: osType,
        },
        timeout: 180000,
      });
      const payload = response?.data ?? response;
      if (payload?.status !== "success") {
        throw new Error(payload?.message || "The comparison did not return a successful result.");
      }
      setComparisonResult(payload);
    } catch (error) {
      setComparisonError(
        error.response?.data?.detail ||
          error.message ||
          "Unable to compare cloud VM prices."
      );
    } finally {
      setIsComparingClouds(false);
    }
  };

  const handleUseComparedVm = (option) => {
    setProvider(option.provider);
    setMachineType(option.sku);
    setVcpu(String(option.vcpu));
    setMemoryGb(String(option.ram_gb));
    setRegion(option.region);
    setOsType(String(option.os_type || "linux").toLowerCase());
    setCurrency(option.currency || currency);
    setProposedMonthlyCost(String(option.monthly_compute_estimate));
    setScenarioName(`${option.provider} ${option.sku} cross-cloud comparison`);
    setScenarioSuccess("VM details copied into the cost scenario form. Review the scope before saving.");
  };

  return (
    <section className="mt-8 overflow-hidden rounded-3xl border border-slate-200 bg-white shadow-sm">
      <div className="border-b border-slate-200 bg-gradient-to-r from-emerald-50 via-white to-blue-50 p-7">
        <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <div className="flex items-center gap-2">
              <CircleDollarSign size={22} className="text-emerald-600" />
              <h2 className="text-xl font-bold text-slate-800">
                Cost Scenario Engine
              </h2>
            </div>
            <p className="mt-2 text-sm text-slate-500">
              Compare a proposed resource configuration and monthly cost for{" "}
              <span className="font-semibold text-slate-700">
                {application.name}
              </span>
              .
            </p>
          </div>
          <button
            type="button"
            onClick={loadScenarios}
            disabled={isLoadingScenarios}
            className="inline-flex items-center justify-center gap-2 rounded-xl border border-slate-200 bg-white px-4 py-2.5 text-sm font-semibold text-slate-700 hover:bg-slate-50 disabled:opacity-60"
          >
            <RefreshCw
              size={16}
              className={isLoadingScenarios ? "animate-spin" : ""}
            />
            Refresh scenarios
          </button>
        </div>
      </div>

      <div className="border-b border-slate-200 bg-slate-50/70 p-7">
        <div className="mb-4">
          <h3 className="text-base font-bold text-slate-800">Unified resource + cost decision</h3>
          <p className="mt-1 text-xs text-slate-500">
            The resource engine determines whether a change is appropriate; the cost engine estimates its financial impact.
          </p>
        </div>
        <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
          <div className="rounded-xl border border-slate-200 bg-white p-4">
            <p className="text-[10px] font-bold uppercase tracking-wide text-slate-400">Latest resource optimizer decision</p>
            <p className="mt-2 text-lg font-extrabold text-slate-800">
              {latestResourceAction || "Analysis not run"}
            </p>
            <p className="mt-1 text-xs leading-relaxed text-slate-500">
              {latestResourceAction === "KEEP"
                ? "Maintain the current configuration. A lower price does not authorize a resize."
                : latestResourceAction === "RESIZE"
                  ? "A resize is suggested by the resource engine; validate target capacity and performance before applying."
                  : latestResourceAction === "SWITCH"
                    ? "A switch is suggested; compatibility, security, migration, downtime and rollback checks are still required."
                    : "Run workload resource analysis to obtain the latest KEEP, RESIZE or SWITCH decision."}
            </p>
          </div>
          <div className="rounded-xl border border-slate-200 bg-white p-4">
            <p className="text-[10px] font-bold uppercase tracking-wide text-slate-400">Latest saved scenario safety</p>
            <p className="mt-2 text-sm font-extrabold text-slate-800">
              {latestScenarioSafety.status || "No saved scenario assessed"}
            </p>
            <p className="mt-1 text-xs leading-relaxed text-slate-500">
              {latestScenarioSafety.reasons?.[0] || "Save a cost scenario to see its workload safety assessment here."}
            </p>
          </div>
        </div>
        <p className="mt-3 text-[10px] text-slate-400">
          Advisory only: no cloud resources are changed automatically. Cost estimates may cover VM compute only and must not be treated as total infrastructure savings.
        </p>
      </div>

      <div className="border-b border-slate-200 p-7">
        <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <div className="flex items-center gap-2">
              <BrainCircuit size={20} className="text-violet-600" />
              <h3 className="text-base font-bold text-slate-800">
                CloudMorph Final Recommendation
              </h3>
            </div>
            <p className="mt-1 text-xs text-slate-500">
              Combined decision from the forecast-aware Resource Optimizer and Cost Scenario Engine.
            </p>
          </div>
          <button
            type="button"
            onClick={loadFinalRecommendation}
            disabled={isLoadingFinalRecommendation}
            className="inline-flex items-center justify-center gap-2 rounded-xl border border-slate-200 bg-white px-4 py-2.5 text-xs font-semibold text-slate-700 hover:bg-slate-50 disabled:cursor-not-allowed disabled:opacity-60"
          >
            <RefreshCw size={14} className={isLoadingFinalRecommendation ? "animate-spin" : ""} />
            Refresh final decision
          </button>
        </div>

        {isLoadingFinalRecommendation && (
          <div className="mt-4 rounded-xl border border-slate-200 bg-slate-50 p-4 text-xs text-slate-500">
            Loading the latest final recommendation...
          </div>
        )}

        {!isLoadingFinalRecommendation && finalRecommendationError && (
          <div className="mt-4 rounded-xl border border-amber-200 bg-amber-50 p-4 text-xs leading-relaxed text-amber-800">
            {finalRecommendationError}
          </div>
        )}

        {!isLoadingFinalRecommendation && finalRecommendation?.final_recommendation && (() => {
          const final = finalRecommendation.final_recommendation;
          const context = finalRecommendation.source_context || {};
          const safety = finalRecommendation.safety || {};
          const action = String(final.action || "HOLD").toUpperCase();
          const actionStyle = action === "KEEP"
            ? "border-emerald-200 bg-emerald-50 text-emerald-800"
            : action === "RESIZE"
              ? "border-blue-200 bg-blue-50 text-blue-800"
              : action === "SWITCH"
                ? "border-violet-200 bg-violet-50 text-violet-800"
                : "border-amber-200 bg-amber-50 text-amber-800";
          const savings = toFiniteNumber(final.potential_monthly_savings, null);

          return (
            <div className="mt-5 rounded-2xl border border-slate-200 bg-slate-50 p-5">
              <div className="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
                <div>
                  <p className="text-[10px] font-bold uppercase tracking-wider text-slate-400">
                    Final action · {final.priority || "Priority not set"}
                  </p>
                  <h4 className="mt-1 text-2xl font-extrabold text-slate-900">
                    {action}
                  </h4>
                  <p className="mt-1 text-sm font-semibold text-slate-700">
                    {final.title || "Final recommendation"}
                  </p>
                  <p className="mt-2 text-xs leading-relaxed text-slate-500">
                    {final.status || "Status not available"}
                  </p>
                </div>
                <span className={`inline-flex w-fit rounded-full border px-3 py-1.5 text-[10px] font-bold ${actionStyle}`}>
                  {action === "HOLD" ? "VALIDATION REQUIRED" : action}
                </span>
              </div>

              <div className="mt-5 grid grid-cols-1 gap-3 sm:grid-cols-3">
                <div className="rounded-xl border border-slate-200 bg-white p-3">
                  <p className="text-[10px] text-slate-400">Resource optimizer</p>
                  <p className="mt-1 text-sm font-bold text-slate-800">
                    {final.resource_optimizer_action || "Not available"}
                  </p>
                </div>
                <div className="rounded-xl border border-slate-200 bg-white p-3">
                  <p className="text-[10px] text-slate-400">Potential monthly savings</p>
                  <p className="mt-1 text-sm font-bold text-slate-800">
                    {savings === null ? "Not established" : formatMoney(savings, "USD")}
                  </p>
                </div>
                <div className="rounded-xl border border-slate-200 bg-white p-3">
                  <p className="text-[10px] text-slate-400">Provider pricing</p>
                  <p className="mt-1 text-sm font-bold text-slate-800">
                    {final.savings_are_provider_verified ? "Verified" : "Not verified"}
                  </p>
                </div>
              </div>

              {Array.isArray(final.reasons) && final.reasons.length > 0 && (
                <div className="mt-5">
                  <p className="text-xs font-bold text-slate-700">Why CloudMorph made this decision</p>
                  <ul className="mt-2 space-y-2">
                    {final.reasons.map((reason, index) => (
                      <li key={`${index}-${reason}`} className="flex gap-2 text-xs leading-relaxed text-slate-600">
                        <CheckCircle2 size={14} className="mt-0.5 shrink-0 text-blue-500" />
                        <span>{reason}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {Array.isArray(final.required_validation) && final.required_validation.length > 0 && (
                <div className="mt-5 rounded-xl border border-amber-200 bg-amber-50 p-4">
                  <p className="flex items-center gap-2 text-xs font-bold text-amber-800">
                    <ShieldAlert size={15} />
                    Required validation before action
                  </p>
                  <ul className="mt-2 space-y-2">
                    {final.required_validation.map((item, index) => (
                      <li key={`${index}-${item}`} className="text-xs leading-relaxed text-amber-800">
                        • {item}
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              <div className="mt-4 flex flex-wrap gap-x-5 gap-y-2 border-t border-slate-200 pt-4 text-[10px] text-slate-500">
                <span>Workload source: {context.workload_source || "Not available"}</span>
                <span>Cost scenario: {final.cost_scenario_usable ? "Usable" : "Not usable / unavailable"}</span>
                <span>Scenario matches decision: {context.cost_scenario_action_matches ? "Yes" : "No"}</span>
                <span>Scenario fresh: {context.cost_scenario_fresh_for_latest_resource_decision ? "Yes" : "No"}</span>
              </div>

              <div className="mt-3 flex items-start gap-2 rounded-lg bg-white p-3 text-[10px] leading-relaxed text-slate-500">
                <ShieldCheck size={14} className="mt-0.5 shrink-0 text-emerald-600" />
                <span>
                  Manual validation required: {safety.manual_validation_required ? "Yes" : "No"}.
                  Automatic infrastructure changes: {safety.automatic_infrastructure_changes ? "Enabled" : "Disabled"}.
                  The orchestrator does not apply cloud changes automatically.
                </span>
              </div>
            </div>
          );
        })()}
      </div>

      <div className="border-b border-slate-200 p-7">
        <div className="mb-5">
          <h3 className="text-base font-bold text-slate-800">Cross-cloud VM cost comparison</h3>
          <p className="mt-1 text-sm text-slate-500">
            Compare suitable AWS, Azure and GCP virtual machines using public on-demand compute prices.
          </p>
        </div>

        <form onSubmit={handleCompareClouds} className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-4">
          <div>
            <label className="mb-1.5 block text-xs font-semibold text-slate-600">Required vCPU</label>
            <input type="number" min="1" max="256" step="1" required value={compareVcpu}
              onChange={(event) => setCompareVcpu(event.target.value)}
              className="w-full rounded-xl border border-slate-200 px-3 py-2.5 text-sm" />
          </div>
          <div>
            <label className="mb-1.5 block text-xs font-semibold text-slate-600">Minimum RAM (GB)</label>
            <input type="number" min="0.1" max="4096" step="0.1" required value={compareRamGb}
              onChange={(event) => setCompareRamGb(event.target.value)}
              className="w-full rounded-xl border border-slate-200 px-3 py-2.5 text-sm" />
          </div>
          <div>
            <label className="mb-1.5 block text-xs font-semibold text-slate-600">AWS region</label>
            <input required value={awsRegion} onChange={(event) => setAwsRegion(event.target.value)}
              className="w-full rounded-xl border border-slate-200 px-3 py-2.5 text-sm" />
          </div>
          <div>
            <label className="mb-1.5 block text-xs font-semibold text-slate-600">Azure region</label>
            <input required value={azureRegion} onChange={(event) => setAzureRegion(event.target.value)}
              className="w-full rounded-xl border border-slate-200 px-3 py-2.5 text-sm" />
          </div>
          <div>
            <label className="mb-1.5 block text-xs font-semibold text-slate-600">GCP region</label>
            <input required value={gcpRegion} onChange={(event) => setGcpRegion(event.target.value)}
              className="w-full rounded-xl border border-slate-200 px-3 py-2.5 text-sm" />
          </div>
          <div>
            <label className="mb-1.5 block text-xs font-semibold text-slate-600">Operating system</label>
            <select value={osType} onChange={(event) => setOsType(event.target.value)}
              className="w-full rounded-xl border border-slate-200 bg-white px-3 py-2.5 text-sm">
              <option value="linux">Linux</option>
              <option value="windows">Windows</option>
            </select>
          </div>
          <div>
            <label className="mb-1.5 block text-xs font-semibold text-slate-600">Currency</label>
            <select value={currency} onChange={(event) => setCurrency(event.target.value)}
              className="w-full rounded-xl border border-slate-200 bg-white px-3 py-2.5 text-sm">
              <option value="USD">USD</option>
            </select>
          </div>
          <div className="flex items-end">
            <button type="submit" disabled={isComparingClouds}
              className="inline-flex w-full items-center justify-center gap-2 rounded-xl bg-slate-900 px-4 py-2.5 text-sm font-semibold text-white hover:bg-slate-700 disabled:cursor-not-allowed disabled:opacity-60">
              {isComparingClouds ? "Comparing prices..." : "Compare cloud VMs"}
            </button>
          </div>
        </form>

        {comparisonError && (
          <div role="alert" className="mt-4 rounded-xl border border-red-200 bg-red-50 p-3 text-sm text-red-700">{comparisonError}</div>
        )}

        {comparisonResult && (
          <div className="mt-6 space-y-4">
            <div className="rounded-2xl border border-emerald-200 bg-emerald-50 p-4">
              <p className="text-xs font-semibold uppercase tracking-wide text-emerald-700">Cheapest suitable option</p>
              {comparisonResult.cheapest_option ? (
                <div className="mt-2 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
                  <div>
                    <p className="text-lg font-bold text-slate-900">
                      {comparisonResult.cheapest_option.provider} · {comparisonResult.cheapest_option.sku}
                    </p>
                    <p className="text-sm text-slate-600">
                      {comparisonResult.cheapest_option.vcpu} vCPU · {comparisonResult.cheapest_option.ram_gb} GB RAM · {comparisonResult.cheapest_option.region}
                    </p>
                  </div>
                  <div className="sm:text-right">
                    <p className="text-xl font-bold text-emerald-800">
                      {formatMoney(comparisonResult.cheapest_option.monthly_compute_estimate, comparisonResult.cheapest_option.currency)}/mo
                    </p>
                    <p className="text-xs text-slate-500">
                      {formatHourlyMoney(comparisonResult.cheapest_option.hourly_price, comparisonResult.cheapest_option.currency)}/hour
                    </p>
                  </div>
                  <button type="button" onClick={() => handleUseComparedVm(comparisonResult.cheapest_option)}
                    className="rounded-xl bg-emerald-700 px-4 py-2 text-sm font-semibold text-white hover:bg-emerald-800">
                    Use in scenario
                  </button>
                </div>
              ) : (
                <p className="mt-2 text-sm text-slate-700">No suitable priced VM was found for these requirements.</p>
              )}
            </div>

            {comparisonResult.cheapest_option && (
              <div className="rounded-2xl border border-blue-100 bg-blue-50/60 p-4">
                <div className="flex items-start gap-2">
                  <Lightbulb size={18} className="mt-0.5 shrink-0 text-blue-700" />
                  <div>
                    <p className="text-sm font-bold text-slate-800">Why this VM is recommended</p>
                    <p className="mt-1 text-sm text-slate-600">
                      It has the lowest verified monthly compute price among the returned candidates while meeting your minimum requirement of {comparisonResult.requirements?.required_vcpu ?? compareVcpu} vCPU and {comparisonResult.requirements?.required_ram_gb ?? compareRamGb} GB RAM.
                      {` `}
                      It provides {Math.max(0, Number(comparisonResult.cheapest_option.vcpu || 0) - Number(comparisonResult.requirements?.required_vcpu ?? compareVcpu))} additional vCPU and {Math.max(0, Number(comparisonResult.cheapest_option.ram_gb || 0) - Number(comparisonResult.requirements?.required_ram_gb ?? compareRamGb)).toFixed(1)} GB additional RAM above the requested minimum.
                    </p>
                    <p className="mt-1 text-xs text-slate-500">Recommendation is based on public VM compute pricing and resource fit only; it does not assess workload performance, availability, licensing, storage or network costs.</p>
                  </div>
                </div>
              </div>
            )}

            <div className="grid grid-cols-1 gap-3 md:grid-cols-3">
              {Object.entries((comparisonResult.options || []).reduce((groups, option) => {
                const key = option.provider || "Other";
                if (!groups[key]) groups[key] = { count: 0, cheapest: null };
                groups[key].count += 1;
                if (!groups[key].cheapest || Number(option.monthly_compute_estimate) < Number(groups[key].cheapest.monthly_compute_estimate)) {
                  groups[key].cheapest = option;
                }
                return groups;
              }, {})).map(([providerName, summary]) => (
                <div key={providerName} className="rounded-xl border border-slate-200 bg-white p-3">
                  <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">{providerName}</p>
                  <p className="mt-1 text-lg font-bold text-slate-800">{formatMoney(summary.cheapest.monthly_compute_estimate, summary.cheapest.currency)}/mo</p>
                  <p className="text-xs text-slate-500">Lowest returned VM · {summary.count} option{summary.count === 1 ? "" : "s"}</p>
                  <p className="mt-1 text-xs text-slate-600">{summary.cheapest.sku} · {formatHourlyMoney(summary.cheapest.hourly_price, summary.cheapest.currency)}/hour</p>
                </div>
              ))}
            </div>

            <div className="overflow-x-auto rounded-xl border border-slate-200">
              <table className="min-w-full divide-y divide-slate-200 text-left text-sm">
                <thead className="bg-slate-50 text-xs uppercase text-slate-500">
                  <tr>
                    <th className="px-4 py-3">Provider / VM</th>
                    <th className="px-4 py-3">Resources</th>
                    <th className="px-4 py-3">Hourly</th>
                    <th className="px-4 py-3">Monthly</th>
                    <th className="px-4 py-3">Annual</th>
                    <th className="px-4 py-3">Action</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100 bg-white">
                  {(comparisonResult.options || []).map((option, index) => (
                    <tr key={`${option.provider}-${option.sku}-${option.region}-${index}`}
                      className={comparisonResult.cheapest_option?.provider === option.provider && comparisonResult.cheapest_option?.sku === option.sku ? "bg-emerald-50/60" : ""}>
                      <td className="whitespace-nowrap px-4 py-3">
                        <p className="font-semibold text-slate-800">{option.provider}</p>
                        <p className="text-xs text-slate-500">{option.sku}</p>
                        <p className="text-xs text-slate-400">{option.region}</p>
                      </td>
                      <td className="whitespace-nowrap px-4 py-3 text-slate-600">{option.vcpu} vCPU · {option.ram_gb} GB</td>
                      <td className="whitespace-nowrap px-4 py-3 text-slate-600">{formatHourlyMoney(option.hourly_price, option.currency)}</td>
                      <td className="whitespace-nowrap px-4 py-3 font-semibold text-slate-800">{formatMoney(option.monthly_compute_estimate, option.currency)}</td>
                      <td className="whitespace-nowrap px-4 py-3 text-slate-600">{formatMoney(option.annual_compute_estimate, option.currency)}</td>
                      <td className="whitespace-nowrap px-4 py-3">
                        <button type="button" onClick={() => handleUseComparedVm(option)}
                          className="rounded-lg border border-slate-300 px-3 py-1.5 text-xs font-semibold text-slate-700 hover:bg-slate-50">
                          Use in scenario
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="text-xs text-slate-500">
              {comparisonResult.pricing_disclaimer || "Public list-price estimates; actual billing may differ."}
              {" "}Prices use {comparisonResult.cheapest_option?.hours_per_estimated_month || 730} hours per estimated month.
            </p>
            {Array.isArray(comparisonResult.unavailable_candidates) && comparisonResult.unavailable_candidates.length > 0 && (
              <details className="rounded-xl border border-slate-200 p-3 text-sm">
                <summary className="cursor-pointer font-semibold text-slate-700">
                  {comparisonResult.unavailable_candidates.length} unavailable candidates
                </summary>
                <ul className="mt-2 list-disc space-y-1 pl-5 text-xs text-slate-500">
                  {comparisonResult.unavailable_candidates.map((item, index) => (
                    <li key={`${item.provider}-${item.sku}-${index}`}>{item.provider} {item.sku}: {item.message || item.reason}</li>
                  ))}
                </ul>
              </details>
            )}
          </div>
        )}
      </div>

      <div className="grid grid-cols-1 gap-7 p-7 xl:grid-cols-2">
        <form onSubmit={handleCreateScenario} className="space-y-4">
          <div>
            <h3 className="text-base font-bold text-slate-800">
              Create a cost scenario
            </h3>
            <p className="mt-1 text-xs text-slate-500">
              Fetch public provider pricing automatically or enter your own proposed monthly quote.
            </p>
          </div>

          <div>
            <label className="mb-1.5 block text-xs font-semibold text-slate-600">
              Scenario name
            </label>
            <input
              value={scenarioName}
              onChange={(event) => setScenarioName(event.target.value)}
              maxLength={120}
              placeholder="e.g. Smaller application configuration"
              className="w-full rounded-xl border border-slate-200 px-3 py-2.5 text-sm outline-none focus:border-blue-400 focus:ring-4 focus:ring-blue-100"
            />
          </div>

          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
            <div>
              <label className="mb-1.5 block text-xs font-semibold text-slate-600">
                Cloud provider
              </label>
              <select
                value={provider}
                onChange={(event) => {
                  setProvider(event.target.value);
                  setProviderPricing(null);
                }}
                className="w-full rounded-xl border border-slate-200 bg-white px-3 py-2.5 text-sm outline-none focus:border-blue-400 focus:ring-4 focus:ring-blue-100"
              >
                <option value="AZURE">Microsoft Azure</option>
                <option value="AWS">Amazon Web Services</option>
                <option value="GCP">Google Cloud Platform</option>
              </select>
            </div>
            <div>
              <label className="mb-1.5 block text-xs font-semibold text-slate-600">
                Operating system
              </label>
              <select
                value={osType}
                onChange={(event) => {
                  setOsType(event.target.value);
                  setProviderPricing(null);
                }}
                className="w-full rounded-xl border border-slate-200 bg-white px-3 py-2.5 text-sm outline-none focus:border-blue-400 focus:ring-4 focus:ring-blue-100"
              >
                <option value="linux">Linux</option>
                <option value="windows">Windows</option>
              </select>
            </div>
            <div>
              <label className="mb-1.5 block text-xs font-semibold text-slate-600">
                Machine / instance type / SKU
              </label>
              <input
                value={machineType}
                onChange={(event) => {
                  setMachineType(event.target.value);
                  setProviderPricing(null);
                }}
                placeholder="e.g. Standard_B2s"
                className="w-full rounded-xl border border-slate-200 px-3 py-2.5 text-sm outline-none focus:border-blue-400 focus:ring-4 focus:ring-blue-100"
              />
            </div>
            <div>
              <label className="mb-1.5 block text-xs font-semibold text-slate-600">
                Region
              </label>
              <input
                value={region}
                onChange={(event) => {
                  setRegion(event.target.value);
                  setProviderPricing(null);
                }}
                placeholder="e.g. centralindia"
                className="w-full rounded-xl border border-slate-200 px-3 py-2.5 text-sm outline-none focus:border-blue-400 focus:ring-4 focus:ring-blue-100"
              />
            </div>
            <div>
              <label className="mb-1.5 block text-xs font-semibold text-slate-600">
                vCPU
              </label>
              <input
                type="number"
                min="0.1"
                step="0.1"
                value={vcpu}
                onChange={(event) => setVcpu(event.target.value)}
                placeholder="e.g. 2"
                className="w-full rounded-xl border border-slate-200 px-3 py-2.5 text-sm outline-none focus:border-blue-400 focus:ring-4 focus:ring-blue-100"
              />
            </div>
            <div>
              <label className="mb-1.5 block text-xs font-semibold text-slate-600">
                Memory (GB)
              </label>
              <input
                type="number"
                min="0.1"
                step="0.1"
                value={memoryGb}
                onChange={(event) => setMemoryGb(event.target.value)}
                placeholder="e.g. 4"
                className="w-full rounded-xl border border-slate-200 px-3 py-2.5 text-sm outline-none focus:border-blue-400 focus:ring-4 focus:ring-blue-100"
              />
            </div>
            <div>
              <label className="mb-1.5 block text-xs font-semibold text-slate-600">
                Storage (GB)
              </label>
              <input
                type="number"
                min="0.1"
                step="0.1"
                value={storageGb}
                onChange={(event) => setStorageGb(event.target.value)}
                placeholder="e.g. 50"
                className="w-full rounded-xl border border-slate-200 px-3 py-2.5 text-sm outline-none focus:border-blue-400 focus:ring-4 focus:ring-blue-100"
              />
            </div>
            <div>
              <label className="mb-1.5 block text-xs font-semibold text-slate-600">
                Currency
              </label>
              <select
                value={currency}
                onChange={(event) => setCurrency(event.target.value)}
                className="w-full rounded-xl border border-slate-200 bg-white px-3 py-2.5 text-sm outline-none focus:border-blue-400 focus:ring-4 focus:ring-blue-100"
              >
                <option value="USD">USD ($)</option>
                <option value="INR">INR (₹)</option>
                <option value="EUR">EUR (€)</option>
                <option value="GBP">GBP (£)</option>
                <option value="AUD">AUD ($)</option>
                <option value="CAD">CAD ($)</option>
                <option value="SGD">SGD ($)</option>
                <option value="JPY">JPY (¥)</option>
              </select>
            </div>
          </div>

          <button
            type="button"
            onClick={handleFetchProviderPricing}
            disabled={isFetchingPricing || isSavingScenario}
            className="inline-flex w-full items-center justify-center gap-2 rounded-xl border border-blue-200 bg-blue-50 px-4 py-3 text-sm font-bold text-blue-700 transition hover:bg-blue-100 disabled:cursor-not-allowed disabled:opacity-60"
          >
            {isFetchingPricing ? (
              <RefreshCw size={16} className="animate-spin" />
            ) : (
              <Cloud size={16} />
            )}
            {isFetchingPricing ? "Fetching provider pricing..." : "Fetch verified provider pricing"}
          </button>

          {providerPricing?.status === "success" && providerPricing?.pricing_verified === true && (
            <div className="rounded-xl border border-emerald-200 bg-emerald-50 p-4">
              <div className="flex items-center justify-between gap-2">
                <p className="text-xs font-bold text-emerald-800">
                  Verified public {providerPricing.provider} retail pricing
                </p>
                <span className="rounded-full bg-emerald-100 px-2 py-1 text-[10px] font-bold text-emerald-700">
                  VERIFIED
                </span>
              </div>
              <div className="mt-3 grid grid-cols-3 gap-2">
                <div>
                  <p className="text-[10px] text-emerald-700">Hourly</p>
                  <p className="mt-1 text-sm font-bold text-emerald-900">
                    {formatMoney(providerPricing.hourly_price, providerPricing.currency)}
                  </p>
                </div>
                <div>
                  <p className="text-[10px] text-emerald-700">Monthly</p>
                  <p className="mt-1 text-sm font-bold text-emerald-900">
                    {formatMoney(providerPricing.monthly_compute_estimate, providerPricing.currency)}
                  </p>
                </div>
                <div>
                  <p className="text-[10px] text-emerald-700">Annual</p>
                  <p className="mt-1 text-sm font-bold text-emerald-900">
                    {formatMoney(providerPricing.annual_compute_estimate, providerPricing.currency)}
                  </p>
                </div>
              </div>
              <p className="mt-2 text-[10px] leading-relaxed text-emerald-800">
                {providerPricing.sku} · {providerPricing.region} · {providerPricing.os_type} · {providerPricing.unit_of_measure}
              </p>
              <p className="mt-1 text-[10px] leading-relaxed text-emerald-800">
                Compute estimate only. Storage, networking, taxes, discounts and other services are excluded.
              </p>
              {providerPricing.source && (
                <a
                  href={providerPricing.source}
                  target="_blank"
                  rel="noreferrer"
                  className="mt-2 inline-block text-[10px] font-semibold text-emerald-800 underline"
                >
                  View pricing source
                </a>
              )}
            </div>
          )}

          <div>
            <label className="mb-1.5 block text-xs font-semibold text-slate-600">
              Proposed monthly cost ({currency}) — optional when provider pricing is available
            </label>
            <input
              type="number"
              min="0.01"
              step="0.01"
              value={proposedMonthlyCost}
              onChange={(event) => setProposedMonthlyCost(event.target.value)}
              placeholder="Leave blank to use provider pricing automatically"
              className="w-full rounded-xl border border-slate-200 px-3 py-2.5 text-sm outline-none focus:border-blue-400 focus:ring-4 focus:ring-blue-100"
            />
            <p className="mt-1.5 text-[11px] text-slate-400">
              Enter a manual quote to override provider pricing. Leave blank to let the backend fetch the selected provider's public price when the scenario is evaluated.
            </p>
          </div>

          {scenarioError && (
            <div className="rounded-xl border border-red-200 bg-red-50 p-3 text-xs text-red-700">
              {scenarioError}
            </div>
          )}
          {scenarioSuccess && (
            <div className="rounded-xl border border-emerald-200 bg-emerald-50 p-3 text-xs text-emerald-700">
              {scenarioSuccess}
            </div>
          )}

          <button
            type="submit"
            disabled={isSavingScenario}
            className="inline-flex w-full items-center justify-center gap-2 rounded-xl bg-slate-900 px-5 py-3 text-sm font-bold text-white transition hover:bg-blue-600 disabled:cursor-not-allowed disabled:opacity-60"
          >
            {isSavingScenario ? (
              <RefreshCw size={17} className="animate-spin" />
            ) : (
              <Sparkles size={17} />
            )}
            {isSavingScenario ? "Evaluating scenario..." : "Evaluate & Save Scenario"}
          </button>
        </form>

        <div>
          <div className="flex items-center justify-between gap-3">
            <div>
              <h3 className="text-base font-bold text-slate-800">
                Saved cost scenarios
              </h3>
              <p className="mt-1 text-xs text-slate-500">
                {scenarios.length} saved for this application
              </p>
            </div>
            {isLoadingScenarios && (
              <span className="text-xs text-slate-400">Loading...</span>
            )}
          </div>

          {scenarioError && (
            <div className="mt-3 rounded-xl border border-red-200 bg-red-50 px-3 py-2 text-xs text-red-700">
              {scenarioError}
            </div>
          )}
          {scenarioSuccess && (
            <div className="mt-3 rounded-xl border border-emerald-200 bg-emerald-50 px-3 py-2 text-xs text-emerald-700">
              {scenarioSuccess}
            </div>
          )}

          <div className="mt-4 space-y-3">
            {!isLoadingScenarios && scenarios.length === 0 && (
              <div className="rounded-2xl border border-dashed border-slate-300 bg-slate-50 p-8 text-center">
                <Database size={25} className="mx-auto text-slate-400" />
                <p className="mt-3 text-sm font-semibold text-slate-600">
                  No cost scenarios yet
                </p>
                <p className="mt-1 text-xs text-slate-400">
                  Create your first scenario to see its cost comparison here.
                </p>
              </div>
            )}

            {scenarios.map((item, index) => {
              const comparison = item.cost_comparison || {};
              const safety = item.safety_assessment || {};
              const scenario = item.scenario || {};
              const monthlySavings = toFiniteNumber(
                comparison.estimated_monthly_savings,
                null
              );
              const scenarioCurrency = scenario.currency || currency;
              const scenarioId = item.scenario_id || `${index}`;

              return (
                <div
                  key={scenarioId}
                  className="rounded-2xl border border-slate-200 bg-white p-4"
                >
                  <div className="flex flex-wrap items-start justify-between gap-2">
                    <div>
                      <p className="text-sm font-bold text-slate-800">
                        {scenario.name || "Cost scenario"}
                      </p>
                      <p className="mt-1 text-[11px] text-slate-400">
                        {scenario.created_at
                          ? new Date(scenario.created_at).toLocaleString()
                          : "Date unavailable"}
                      </p>
                    </div>
                    <div className="flex items-center gap-2">
                      <span
                        className={`rounded-full px-2.5 py-1 text-[10px] font-bold ${
                          comparison.comparison_status === "SAVINGS"
                            ? "bg-emerald-50 text-emerald-700"
                            : comparison.comparison_status === "COST_INCREASE"
                              ? "bg-amber-50 text-amber-700"
                              : "bg-slate-100 text-slate-600"
                        }`}
                      >
                        {comparison.comparison_status || "NOT COMPARED"}
                      </span>
                      <button
                        type="button"
                        onClick={() => handleDeleteScenario(item.scenario_id, scenario.name)}
                        disabled={!item.scenario_id || deletingScenarioId !== null}
                        title="Delete saved scenario"
                        aria-label={`Delete ${scenario.name || "cost scenario"}`}
                        className="inline-flex h-8 w-8 items-center justify-center rounded-lg border border-red-100 text-red-600 transition hover:bg-red-50 disabled:cursor-not-allowed disabled:opacity-40"
                      >
                        {deletingScenarioId === item.scenario_id ? (
                          <RefreshCw size={14} className="animate-spin" />
                        ) : (
                          <Trash2 size={14} />
                        )}
                      </button>
                    </div>
                  </div>

                  <div className="mt-4 grid grid-cols-2 gap-3">
                    <div className="rounded-xl bg-slate-50 p-3">
                      <p className="text-[10px] text-slate-400">Current / month</p>
                      <p className="mt-1 text-sm font-bold text-slate-700">
                        {formatMoney(comparison.current_monthly_cost, scenarioCurrency)}
                      </p>
                    </div>
                    <div className="rounded-xl bg-slate-50 p-3">
                      <p className="text-[10px] text-slate-400">Proposed / month</p>
                      <p className="mt-1 text-sm font-bold text-slate-700">
                        {formatMoney(comparison.proposed_monthly_cost, scenarioCurrency)}
                      </p>
                    </div>
                  </div>

                  <div className="mt-3 flex flex-wrap items-center justify-between gap-2">
                    <div>
                      <p className="text-[10px] text-slate-400">Estimated monthly savings</p>
                      <p className={`mt-1 text-base font-bold ${
                        monthlySavings > 0 ? "text-emerald-600" : "text-slate-700"
                      }`}>
                        {formatMoney(monthlySavings, scenarioCurrency)}
                      </p>
                    </div>
                    <div className="text-right">
                      <p className="text-[10px] text-slate-400">Annual estimate</p>
                      <p className="mt-1 text-sm font-bold text-slate-700">
                        {formatMoney(comparison.estimated_annual_savings, scenarioCurrency)}
                      </p>
                    </div>
                  </div>

                  <div className="mt-3 rounded-xl border border-slate-100 bg-slate-50 p-3">
                    <p className="text-[10px] font-bold uppercase tracking-wide text-slate-500">
                      Workload safety: {safety.status || "NOT ASSESSED"}
                    </p>
                    <p className="mt-1 text-[11px] leading-relaxed text-slate-500">
                      {safety.reasons?.[0] ||
                        "Manual validation is required before applying any resource change."}
                    </p>
                  </div>

                  <p className="mt-2 text-[10px] text-slate-400">
                    Pricing source: {item.pricing_metadata?.pricing_source || "Not available"} · Provider pricing verified: {item.pricing_metadata?.provider_pricing_verified ? "Yes" : "No"}
                  </p>
                  {item.pricing_metadata?.provider_pricing && (
                    <div className="mt-2 rounded-lg border border-slate-100 bg-slate-50 p-3">
                      <p className="text-[10px] font-bold text-slate-600">
                        {item.pricing_metadata.provider_pricing.provider || "Provider"} · {item.pricing_metadata.provider_pricing.sku || "VM"} · {item.pricing_metadata.provider_pricing.region || "Region"}
                      </p>
                      <p className="mt-1 text-[10px] text-slate-500">
                        Hourly: {formatMoney(item.pricing_metadata.provider_pricing.hourly_price, item.pricing_metadata.provider_pricing.currency || scenarioCurrency)} · Monthly compute: {formatMoney(item.pricing_metadata.provider_pricing.monthly_compute_estimate, item.pricing_metadata.provider_pricing.currency || scenarioCurrency)}
                      </p>
                      <p className="mt-1 text-[10px] leading-relaxed text-slate-400">
                        {item.pricing_metadata.estimate_scope || "Provider compute estimate only"}.
                        {comparison.comparison_note ? ` ${comparison.comparison_note}` : ""}
                      </p>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </section>
  );
}


export default AIAnalysis;