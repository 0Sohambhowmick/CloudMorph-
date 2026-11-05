import { useCallback, useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import API from "../services/api";

import {
  Activity,
  ArrowUpRight,
  BarChart3,
  BrainCircuit,
  CheckCircle2,
  ChevronDown,
  Cloud,
  Cpu,
  DollarSign,
  Gauge,
  HardDrive,
  Layers3,
  Network,
  RefreshCw,
  Server,
  Sparkles,
  TrendingDown,
  TrendingUp,
  X,
  Zap,
} from "lucide-react";

function providerNamesLabel(applications) {
  const names = [...new Set(
    applications
      .map((app) => String(app.provider || "").trim().toUpperCase())
      .filter(Boolean)
  )];
  return names.length ? names.join(" · ") : "No provider data available";
}

function Analytics() {
  const navigate = useNavigate();

  const [timeRange, setTimeRange] = useState("All Apps");
  const [refreshing, setRefreshing] = useState(false);
  const [lastUpdated, setLastUpdated] = useState("Not loaded yet");
  const [selectedApplication, setSelectedApplication] =
    useState(null);
  const [selectedDecision, setSelectedDecision] =
    useState("ALL");
  const [applications, setApplications] = useState([]);
  const [analyticsData, setAnalyticsData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  /* ========================================================= */
  /* FETCH REAL APPLICATIONS FROM THE BACKEND */
  /* ========================================================= */

  const toNumber = (value, fallback = 0) => {
    if (value === null || value === undefined || value === "") return fallback;
    const number = Number(value);
    return Number.isFinite(number) ? number : fallback;
  };

  const nullableNumber = (value) => {
    if (value === null || value === undefined || value === "") return null;
    const number = Number(value);
    return Number.isFinite(number) ? number : null;
  };

  const formatMoney = (value, digits = 2) =>
    Number.isFinite(value) ? `$${value.toFixed(digits)}` : "N/A";

  const formatPercent = (value, digits = 1) =>
    Number.isFinite(value) ? `${value.toFixed(digits)}%` : "N/A";

  const calculateUtilizationFitScore = (utilization) => {
    if (!Number.isFinite(utilization) || utilization < 0 || utilization > 100) {
      return null;
    }

    if (utilization < 35) return Math.max(0, 100 - (35 - utilization) * 2);
    if (utilization > 85) return Math.max(0, 100 - (utilization - 85) * 2);
    return 100;
  };

  const normalizeApplication = (app, costRecord = {}, savedScenario = null) => {
    const cost = nullableNumber(
      costRecord.current_monthly_cost ??
      app.cost ??
      app.monthly_cost ??
      app.monthlyCost
    );

    const decision = String(
      app.decision ?? app.latest_decision ?? app.recommendation_action ?? "INVESTIGATE"
    ).toUpperCase();

    // KEEP means retain the current configuration. Ignore stale lower
    // optimized_cost values on KEEP records; they are not active savings.
    let optimizedCost = decision === "KEEP" ? cost : null;
    let savings = null;
    let costIncrease = null;
    let hasValidScenario = false;

    // A RESIZE/SWITCH estimate is shown only when a saved scenario exists,
    // its current cost matches this application, and both scenario costs
    // are valid. The overview endpoint alone does not prove this mapping.
    if (["RESIZE", "SWITCH"].includes(decision) && savedScenario) {
      const comparison = savedScenario.cost_comparison ?? {};
      const scenarioCurrent = nullableNumber(comparison.current_monthly_cost);
      const scenarioProposed = nullableNumber(comparison.proposed_monthly_cost);
      const currentMatches =
        cost !== null && scenarioCurrent !== null &&
        Math.abs(cost - scenarioCurrent) <= 0.01;

      if (
        currentMatches && scenarioProposed !== null &&
        scenarioProposed > 0 && scenarioCurrent > 0
      ) {
        optimizedCost = scenarioProposed;
        hasValidScenario = true;
        if (scenarioCurrent > scenarioProposed) {
          savings = scenarioCurrent - scenarioProposed;
        } else if (scenarioProposed > scenarioCurrent) {
          costIncrease = scenarioProposed - scenarioCurrent;
        }
      }
    }

    const hasComparableCosts =
      cost !== null && optimizedCost !== null && cost >= 0 && optimizedCost >= 0;

    const rawUtilization = nullableNumber(app.utilization ?? app.usagePercent);
    const utilization =
      rawUtilization !== null && rawUtilization >= 0 && rawUtilization <= 100
        ? rawUtilization
        : null;

    const servicesValue = Array.isArray(app.services)
      ? app.services.length
      : nullableNumber(app.services) !== null
        ? nullableNumber(app.services)
        : app.service
          ? 1
          : 0;

    return {
      ...app,
      id: String(app.id ?? app._id ?? costRecord.application_id ?? ""),
      name: app.name || costRecord.name || "Unnamed Application",
      provider: app.provider || costRecord.provider || "Unknown",
      cost,
      optimizedCost,
      savings,
      costIncrease,
      utilization,
      services: toNumber(app.serviceCount, servicesValue),
      decision,
      category: app.category || app.service || "Application",
      status: app.status ? String(app.status).toUpperCase() : "ACTIVE",
      health: null,
      overallOptimizationScore: nullableNumber(app.overall_optimization_score),
      optimizationScore: nullableNumber(app.optimization_score) ?? calculateUtilizationFitScore(utilization),
      performance: null,
      hasComparableCosts,
      hasValidScenario,
    };
  };

  const fetchApplications = useCallback(async () => {
    setLoading(true);
    setError("");

    try {
      const [applicationsResponse, analyticsResponse] = await Promise.all([
        API.get("/applications"),
        API.get("/analytics/overview"),
      ]);

      const payload = applicationsResponse.data;
      const analytics = analyticsResponse.data;

      let list = [];

      if (Array.isArray(payload)) {
        list = payload;
      } else if (Array.isArray(payload?.applications)) {
        list = payload.applications;
      } else if (Array.isArray(payload?.data)) {
        list = payload.data;
      } else if (payload && Object.keys(payload).length === 0) {
        list = [];
      } else {
        throw new Error(
          "The backend returned an unexpected applications response."
        );
      }

      if (!analytics || analytics.status !== "success") {
        throw new Error("The analytics API returned an unexpected response.");
      }

      const costRecords = Array.isArray(analytics.application_costs)
        ? analytics.application_costs
        : [];

      const costById = new Map(
        costRecords.map((record) => [String(record.application_id), record])
      );

      const validApplications = list.filter(
        (app) => app && typeof app === "object" && !Array.isArray(app)
      );

      // Load saved cost scenarios only for applications whose latest stored
      // decision is RESIZE or SWITCH. Scenario failures remain unavailable;
      // they must never be replaced with stale application-level cost fields.
      const scenarioEntries = await Promise.all(
        validApplications
          .filter((app) => ["RESIZE", "SWITCH"].includes(
            String(app.decision ?? app.latest_decision ?? app.recommendation_action ?? "").toUpperCase()
          ))
          .map(async (app) => {
            const id = String(app.id ?? app._id ?? "");
            if (!id) return [id, null];
            try {
              const response = await API.get(
                `/optimization/applications/${encodeURIComponent(id)}/cost-scenarios`
              );
              const scenarioPayload = response?.data ?? response;
              const scenarios = Array.isArray(scenarioPayload)
                ? scenarioPayload
                : Array.isArray(scenarioPayload?.scenarios)
                  ? scenarioPayload.scenarios
                  : [];
              const matchingScenarios = scenarios
                .filter((item) => {
                  const comparison = item?.cost_comparison ?? {};
                  const scenarioCurrent = nullableNumber(comparison.current_monthly_cost);
                  const scenarioProposed = nullableNumber(comparison.proposed_monthly_cost);
                  const appCost = nullableNumber(app.cost ?? app.monthly_cost ?? app.monthlyCost);
                  return scenarioCurrent !== null && scenarioProposed !== null &&
                    scenarioCurrent > 0 && scenarioProposed > 0 && appCost !== null &&
                    Math.abs(scenarioCurrent - appCost) <= 0.01;
                })
                .sort((a, b) => {
                  const aDate = new Date(a?.scenario?.created_at ?? a?.created_at ?? 0).getTime();
                  const bDate = new Date(b?.scenario?.created_at ?? b?.created_at ?? 0).getTime();
                  return bDate - aDate;
                });
              return [id, matchingScenarios[0] ?? null];
            } catch (scenarioError) {
              console.warn(`Unable to load saved cost scenarios for application ${id}:`, scenarioError);
              return [id, null];
            }
          })
      );
      const scenarioById = new Map(scenarioEntries);

      const normalizedApplications = validApplications.map((app) => {
        const id = String(app.id ?? app._id ?? "");
        return normalizeApplication(
          app,
          costById.get(id) || {},
          scenarioById.get(id) || null
        );
      });

      setApplications(normalizedApplications);
      setAnalyticsData(analytics);
      setLastUpdated(new Date().toLocaleTimeString());
    } catch (err) {
      console.error("Failed to load analytics:", err);
      setApplications([]);
      setAnalyticsData(null);
      setError(
        err.response?.data?.detail ||
          err.message ||
          "Unable to load analytics. Check that the backend is running and try again."
      );
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, []);

  useEffect(() => {
    fetchApplications();
  }, [fetchApplications]);

  /* ========================================================= */
  /* DYNAMIC ANALYTICS CALCULATIONS */
  /* ========================================================= */


  const applicationsWithCurrentCost = applications.filter(
    (application) => Number.isFinite(application.cost)
  );

  const applicationsWithSavingsEstimate = applications.filter(
    (application) => Number.isFinite(application.savings) && application.savings > 0
  );

  const currentCost = applicationsWithCurrentCost.length
    ? applicationsWithCurrentCost.reduce((total, application) => total + application.cost, 0)
    : null;

  const allCostApplicationsHaveOptimizedCost =
    applicationsWithCurrentCost.length > 0 &&
    applicationsWithCurrentCost.every((application) => Number.isFinite(application.optimizedCost));

  const optimizedCost = allCostApplicationsHaveOptimizedCost
    ? applicationsWithCurrentCost.reduce((total, application) => total + application.optimizedCost, 0)
    : null;

  // Only positive differences from matching saved RESIZE/SWITCH scenarios
  // count as savings. KEEP records and overview aggregates are not used.
  const potentialSavings = applicationsWithSavingsEstimate.length
    ? applicationsWithSavingsEstimate.reduce((total, application) => total + application.savings, 0)
    : null;

  const savingsPercentage =
    Number.isFinite(currentCost) &&
    currentCost > 0 &&
    Number.isFinite(potentialSavings)
      ? (potentialSavings / currentCost) * 100
      : null;

  const utilizationValues = applications
    .map((app) => app.utilization)
    .filter((value) => Number.isFinite(value));

  const averageUtilization = utilizationValues.length
    ? utilizationValues.reduce((total, value) => total + value, 0) / utilizationValues.length
    : null;

  // These are optimization metrics supplied by the backend, not health or
  // runtime performance telemetry. Prefer backend aggregates and fall back
  // to available per-application scores when an aggregate is absent.
  const overallOptimizationScores = applications
    .map((app) => app.overallOptimizationScore)
    .filter((value) => Number.isFinite(value));
  const utilizationFitScores = applications
    .map((app) => app.optimizationScore)
    .filter((value) => Number.isFinite(value));

  const averageOverallOptimizationScore =
    nullableNumber(analyticsData?.optimization_analysis?.average_overall_optimization_score) ??
    (overallOptimizationScores.length
      ? overallOptimizationScores.reduce((sum, value) => sum + value, 0) / overallOptimizationScores.length
      : null);

  const averageUtilizationFitScore =
    nullableNumber(analyticsData?.optimization_analysis?.average_utilization_fit_score) ??
    (utilizationFitScores.length
      ? utilizationFitScores.reduce((sum, value) => sum + value, 0) / utilizationFitScores.length
      : null);

  const keepCount = applications.filter((app) => app.decision === "KEEP").length;
  const resizeCount = applications.filter((app) => app.decision === "RESIZE").length;
  const switchCount = applications.filter((app) => app.decision === "SWITCH").length;
  const applicationsWithCostCount = applications.filter((app) => Number.isFinite(app.cost)).length;
  const uniqueProviderCount = new Set(
    applications.map((app) => String(app.provider || "").trim().toUpperCase()).filter(Boolean)
  ).size;

  const totalServices = applications.reduce((total, app) => total + app.services, 0);

  const applicationsWithCost = applications.filter(
    (app) => Number.isFinite(app.cost)
  );

  const highestCostApplication = applicationsWithCost.length
    ? [...applicationsWithCost].sort((a, b) => b.cost - a.cost)[0]
    : null;

  const highestSavingsApplication = applicationsWithSavingsEstimate.length
    ? [...applicationsWithSavingsEstimate].sort((a, b) => b.savings - a.savings)[0]
    : null;

  const maxUtilization = utilizationValues.length
    ? Math.max(...utilizationValues)
    : null;

  const monthlyData = useMemo(() => applications
    .filter((app) => Number.isFinite(app.cost))
    .map((app) => ({
      application: app.name,
      actual: app.cost,
      optimized: Number.isFinite(app.optimizedCost) ? app.optimizedCost : null,
      utilization: app.utilization,
    })), [applications]);

  const maxMonthlyCost = Math.max(1, ...monthlyData.map((item) => item.actual));

  const resourceMetrics = [
    {
      name: "Utilization",
      value: averageUtilization,
      displayValue: formatPercent(averageUtilization),
      progress: averageUtilization,
      icon: Cpu,
      description: "Average application-level utilization; not live CPU telemetry",
    },
    {
      name: "Optimization Decisions",
      value: applications.length,
      displayValue: `${applications.length}`,
      progress: null,
      icon: BrainCircuit,
      description: `${keepCount} KEEP · ${resizeCount} RESIZE · ${switchCount} SWITCH`,
    },
    {
      name: "Cost Data Coverage",
      value: applicationsWithCostCount,
      displayValue: `${applicationsWithCostCount}/${applications.length}`,
      progress: applications.length ? (applicationsWithCostCount / applications.length) * 100 : 0,
      icon: DollarSign,
      description: "Applications with recorded monthly cost",
    },
    {
      name: "Cloud Provider Coverage",
      value: uniqueProviderCount,
      displayValue: `${uniqueProviderCount}`,
      progress: null,
      icon: Cloud,
      description: providerNamesLabel(applications),
    },
  ];

  const providerNames = [...new Set(applications.map((app) => app.provider))];

  const providerData = providerNames.map((name) => {
    const providerApps = applications.filter((app) => app.provider === name);
    const providerCostApps = providerApps.filter((app) => Number.isFinite(app.cost));
    const cost = providerCostApps.length
      ? providerCostApps.reduce((sum, app) => sum + app.cost, 0)
      : null;

    return {
      name,
      cost,
      percentage: Number.isFinite(currentCost) && currentCost > 0 && Number.isFinite(cost)
        ? Math.round((cost / currentCost) * 100)
        : 0,
      services: providerApps.reduce((sum, app) => sum + app.services, 0),
      icon: Cloud,
      description: `${providerApps.length} application${providerApps.length === 1 ? "" : "s"}`,
    };
  });

  const decisionDefinitions = [
    { label: "KEEP", style: "bg-emerald-500", lightStyle: "bg-emerald-50 text-emerald-600", description: "Existing configuration can be retained.", icon: CheckCircle2 },
    { label: "RESIZE", style: "bg-blue-500", lightStyle: "bg-blue-50 text-blue-600", description: "Adjust resources to better match workload.", icon: Gauge },
    { label: "SWITCH", style: "bg-violet-500", lightStyle: "bg-violet-50 text-violet-600", description: "Consider an alternative service or configuration.", icon: Zap },
    { label: "SCALE_UP", style: "bg-amber-500", lightStyle: "bg-amber-50 text-amber-600", description: "Review capacity for elevated resource pressure.", icon: TrendingUp },
    { label: "INVESTIGATE", style: "bg-slate-500", lightStyle: "bg-slate-100 text-slate-600", description: "Collect more information before deciding.", icon: Activity },
  ];

  const decisionData = decisionDefinitions.map((definition) => {
    const count = applications.filter(
      (app) => app.decision === definition.label
    ).length;

    return {
      ...definition,
      count,
      percentage: applications.length
        ? Math.round((count / applications.length) * 100)
        : 0,
    };
  });

  /* ========================================================= */
  /* FILTERED MONTHLY DATA */
  /* ========================================================= */

  const visibleMonthlyData = useMemo(() => {
    if (timeRange === "7 Apps") return monthlyData.slice(0, 7);
    if (timeRange === "30 Apps") return monthlyData.slice(0, 30);
    return monthlyData;
  }, [timeRange, monthlyData]);

  /* ========================================================= */
  /* FILTERED APPLICATIONS */
  /* ========================================================= */

  const visibleApplications = useMemo(() => {
    if (selectedDecision === "ALL") {
      return applications;
    }

    return applications.filter(
      (application) =>
        application.decision === selectedDecision
    );
  }, [applications, selectedDecision]);

  /* ========================================================= */
  /* HELPERS */
  /* ========================================================= */

  const getUtilizationColor = (value) => {
    if (!Number.isFinite(value)) return "text-slate-400";
    if (value >= 85) {
      return "text-red-500";
    }

    if (value >= 70) {
      return "text-amber-500";
    }

    return "text-emerald-500";
  };

  const getUtilizationBar = (value) => {
    if (!Number.isFinite(value)) return "bg-slate-200";
    if (value >= 85) {
      return "bg-red-500";
    }

    if (value >= 70) {
      return "bg-amber-500";
    }

    return "bg-emerald-500";
  };

  const getDecisionStyle = (decision) => {
    if (decision === "KEEP") {
      return "bg-emerald-50 text-emerald-600 border-emerald-100";
    }

    if (decision === "RESIZE") {
      return "bg-blue-50 text-blue-600 border-blue-100";
    }

    return "bg-violet-50 text-violet-600 border-violet-100";
  };

  const getStatusStyle = (status) => {
    if (String(status).toUpperCase() === "ACTIVE") {
      return "bg-emerald-50 text-emerald-600 border-emerald-100";
    }

    return "bg-amber-50 text-amber-600 border-amber-100";
  };

  /* ========================================================= */
  /* REFRESH ANALYTICS */
  /* ========================================================= */

  const handleRefresh = async () => {
    if (refreshing) return;
    setRefreshing(true);
    await fetchApplications();
  };

  /* ========================================================= */
  /* NAVIGATION */
  /* ========================================================= */

  const handleViewDetails = (application) => {
    navigate(`/app/applications/${encodeURIComponent(application.id)}`);
  };

  const handleViewAnalysis = (application) => {
    navigate(`/app/applications/${encodeURIComponent(application.id)}/analysis`, {
      state: {
        application,
      },
    });
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

        <div className="relative px-8 py-10">
          <div className="flex flex-col gap-6 lg:flex-row lg:items-center lg:justify-between">
            <div>
              <div className="flex flex-wrap items-center gap-3">
                <span className="flex items-center gap-2 rounded-full border border-violet-400/20 bg-violet-500/10 px-4 py-2 text-xs font-bold uppercase tracking-wider text-violet-200">
                  <BarChart3 size={15} />
                  Cloud Analytics
                </span>

                <span className="flex items-center gap-2 text-xs font-semibold text-emerald-300">
                  <span className="h-2 w-2 animate-pulse rounded-full bg-emerald-400" />
                  Application Analytics Loaded
                </span>
              </div>

              <h1 className="mt-6 text-3xl font-bold tracking-tight text-white md:text-5xl">
                Analytics
              </h1>

              <p className="mt-4 max-w-2xl text-sm leading-relaxed text-slate-300 md:text-base">
                Review application cost estimates, utilization data,
                provider distribution, and optimization decisions
                using your CloudMorph records.
              </p>

              <div className="mt-5 flex flex-wrap items-center gap-4 text-xs text-slate-400">
                <span className="flex items-center gap-2">
                  <span className="h-2 w-2 rounded-full bg-emerald-400" />
                  Stored application records
                </span>

                <span>
                  Last updated: {lastUpdated}
                </span>
              </div>
            </div>

            {/* HERO ACTIONS */}
            <div className="flex flex-col gap-3 sm:flex-row">
              {/* APPLICATION COUNT FILTER */}
              <div className="relative">
                <select
                  value={timeRange}
                  onChange={(event) =>
                    setTimeRange(event.target.value)
                  }
                  className="appearance-none rounded-xl border border-white/10 bg-white/10 py-3 pl-4 pr-11 text-sm font-bold text-white outline-none backdrop-blur transition focus:border-blue-400"
                >
                  <option
                    value="7 Apps"
                    className="text-slate-800"
                  >
                    First 7 Applications
                  </option>

                  <option
                    value="30 Apps"
                    className="text-slate-800"
                  >
                    First 30 Applications
                  </option>

                  <option
                    value="All Apps"
                    className="text-slate-800"
                  >
                    All Applications
                  </option>
                </select>

                <ChevronDown
                  size={17}
                  className="pointer-events-none absolute right-4 top-1/2 -translate-y-1/2 text-slate-300"
                />
              </div>

              {/* REFRESH */}
              <button
                type="button"
                onClick={handleRefresh}
                disabled={refreshing}
                className="flex items-center justify-center gap-2 rounded-xl border border-white/10 bg-white/10 px-5 py-3 text-sm font-bold text-white backdrop-blur transition hover:bg-white/15 disabled:cursor-not-allowed disabled:opacity-70"
              >
                <RefreshCw
                  size={17}
                  className={
                    refreshing
                      ? "animate-spin"
                      : ""
                  }
                />

                {refreshing
                  ? "Refreshing..."
                  : "Refresh Analytics"}
              </button>
            </div>
          </div>

          {/* HERO METRICS */}
          <div className="mt-10 grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-4">
            <HeroMetric
              icon={DollarSign}
              label="Current Monthly Cost"
              value={formatMoney(currentCost)}
              description="Across applications with recorded monthly costs"
            />

            <HeroMetric
              icon={TrendingDown}
              label="Estimated Monthly Savings"
              value={formatMoney(potentialSavings)}
              description={`${formatPercent(savingsPercentage)} estimated savings · ${formatMoney(Number.isFinite(potentialSavings) ? potentialSavings * 12 : null)} annualized`}
            />

            <HeroMetric
              icon={Gauge}
              label="Avg Utilization"
              value={Number.isFinite(averageUtilization) ? `${averageUtilization.toFixed(1)}%` : "N/A"}
              description="Average of available application utilization values"
            />

            <HeroMetric
              icon={Layers3}
              label="Applications"
              value={applications.length}
              description={`${totalServices} configured services in application records`}
            />
          </div>
        </div>
      </section>

      {/* ===================================================== */}
      {/* MAIN */}
      {/* ===================================================== */}

      <main className="p-8">
        {loading && (
          <div role="status" className="mb-6 rounded-2xl border border-blue-100 bg-white px-5 py-3 text-sm text-blue-700 shadow-sm">
            Loading analytics from your applications...
          </div>
        )}

        {!loading && error && (
          <div role="alert" className="mb-6 flex flex-wrap items-center justify-between gap-3 rounded-2xl border border-red-200 bg-white px-5 py-4 text-sm text-red-700 shadow-sm">
            <span>{error}</span>
            <button type="button" onClick={handleRefresh} className="rounded-lg bg-red-50 px-4 py-2 font-semibold hover:bg-red-100">Try Again</button>
          </div>
        )}

        {/* =================================================== */}
        {/* COST OVERVIEW */}
        {/* =================================================== */}

        <section className="grid grid-cols-1 gap-6 xl:grid-cols-3">
          <div className="xl:col-span-2 rounded-3xl border border-slate-200 bg-white p-7 shadow-sm">
            <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
              <div>
                <div className="flex items-center gap-2">
                  <TrendingUp
                    size={22}
                    className="text-blue-600"
                  />

                  <h2 className="text-xl font-bold text-slate-800">
                    Cost by Application
                  </h2>
                </div>

                <p className="mt-2 text-sm text-slate-500">
                  Current monthly cost compared with saved scenario estimates. KEEP applications retain their current cost; unavailable scenarios are not invented.
                </p>
              </div>

              <div className="flex items-center gap-4 text-xs">
                <span className="flex items-center gap-2 text-slate-500">
                  <span className="h-2.5 w-2.5 rounded-full bg-blue-500" />
                  Actual
                </span>

                <span className="flex items-center gap-2 text-slate-500">
                  <span className="h-2.5 w-2.5 rounded-full bg-emerald-500" />
                  Scenario / retained cost
                </span>
              </div>
            </div>

            {/* CHART */}
            <div className="mt-8">
              {visibleMonthlyData.length === 0 ? (
                <div className="flex h-72 items-center justify-center rounded-2xl border border-dashed border-slate-200 bg-slate-50 px-6 text-center text-sm text-slate-500">
                  Current monthly cost data is not available yet.
                </div>
              ) : (
              <>
              <div className="flex h-72 items-end gap-3 sm:gap-6">
                {visibleMonthlyData.map((item) => {
                  const actualHeight =
                    (item.actual / maxMonthlyCost) * 100;

                  const optimizedHeight = Number.isFinite(item.optimized)
                    ? (item.optimized / maxMonthlyCost) * 100
                    : 0;

                  return (
                    <div
                      key={`${item.application}-${item.actual}-${item.optimized}`}
                      className="group flex h-full flex-1 items-end justify-center gap-1.5 sm:gap-2"
                    >
                      <div className="flex h-full w-1/2 flex-col justify-end">
                        <div
                          className="relative rounded-t-lg bg-blue-500 transition-all duration-700 hover:bg-blue-600"
                          style={{
                            height: `${actualHeight}%`,
                          }}
                          title={`Current cost: ${formatMoney(item.actual, 2)}`}
                        >
                          <span className="absolute -top-6 left-1/2 hidden -translate-x-1/2 whitespace-nowrap rounded-md bg-slate-900 px-2 py-1 text-[9px] font-bold text-white group-hover:block">
                            {formatMoney(item.actual, 2)}
                          </span>
                        </div>
                      </div>

                      <div className="flex h-full w-1/2 flex-col justify-end">
                        {Number.isFinite(item.optimized) ? (
                          <div
                            className="relative rounded-t-lg bg-emerald-500 transition-all duration-700 hover:bg-emerald-600"
                            style={{ height: `${optimizedHeight}%` }}
                            title={`Scenario / retained cost: ${formatMoney(item.optimized, 2)}`}
                          >
                            <span className="absolute -top-6 left-1/2 hidden -translate-x-1/2 whitespace-nowrap rounded-md bg-slate-900 px-2 py-1 text-[9px] font-bold text-white group-hover:block">
                              {formatMoney(item.optimized, 2)}
                            </span>
                          </div>
                        ) : (
                          <span className="pb-1 text-center text-[9px] font-semibold text-slate-400">N/A</span>
                        )}
                      </div>
                    </div>
                  );
                })}
              </div>

              <div className="mt-3 flex gap-3 sm:gap-6">
                {visibleMonthlyData.map((item) => (
                  <div
                    key={`${item.application}-${item.actual}-${item.optimized}-label`}
                    className="min-w-0 flex-1 truncate text-center text-xs font-medium text-slate-400"
                    title={item.application}
                  >
                    {item.application}
                  </div>
                ))}
              </div>
              </>
              )}
            </div>

            {/* CHART SUMMARY */}
            <div className="mt-6 grid grid-cols-1 gap-3 sm:grid-cols-3">
              <div className="rounded-xl bg-blue-50 px-4 py-3">
                <p className="text-[10px] font-medium text-slate-400">
                  Current
                </p>

                <p className="mt-1 text-sm font-bold text-blue-600">
                  {formatMoney(currentCost)}
                </p>
              </div>

              <div className="rounded-xl bg-emerald-50 px-4 py-3">
                <p className="text-[10px] font-medium text-slate-400">
                  Optimized
                </p>

                <p className="mt-1 text-sm font-bold text-emerald-600">
                  {formatMoney(optimizedCost)}
                </p>
              </div>

              <div className="rounded-xl bg-violet-50 px-4 py-3">
                <p className="text-[10px] font-medium text-slate-400">
                  Savings
                </p>

                <p className="mt-1 text-sm font-bold text-violet-600">
                  {formatMoney(potentialSavings)}
                </p>
              </div>
            </div>
          </div>

          {/* SAVINGS CARD */}
          <div className="rounded-3xl border border-slate-200 bg-white p-7 shadow-sm">
            <div className="flex items-center gap-2">
              <Sparkles
                size={22}
                className="text-violet-600"
              />

              <h2 className="text-xl font-bold text-slate-800">
                Optimization Impact
              </h2>
            </div>

            <p className="mt-2 text-sm text-slate-500">
              Estimated, unverified savings from matching saved RESIZE or SWITCH scenarios. Manual validation is required.
            </p>

            <div className="mt-8 flex items-center justify-center">
              <div className="relative flex h-48 w-48 items-center justify-center rounded-full bg-gradient-to-br from-violet-100 via-blue-50 to-emerald-100">
                <div className="absolute inset-4 flex items-center justify-center rounded-full bg-white shadow-inner">
                  <div className="text-center">
                    <p className="text-4xl font-bold text-slate-800">
                      {formatPercent(savingsPercentage)}
                    </p>

                    <p className="mt-1 text-xs font-medium text-slate-400">
                      Estimated Cost Reduction
                    </p>
                  </div>
                </div>
              </div>
            </div>

            <div className="mt-8 space-y-4">
              <ImpactRow
                label="Current Cost"
                value={formatMoney(currentCost)}
                icon={DollarSign}
              />

              <ImpactRow
                label="Optimized Cost"
                value={formatMoney(optimizedCost)}
                icon={TrendingDown}
              />

              <ImpactRow
                label="Estimated Monthly Savings"
                value={formatMoney(potentialSavings)}
                icon={Sparkles}
                highlight
              />
            </div>
          </div>
        </section>

        {/* =================================================== */}
        {/* PERFORMANCE SNAPSHOT */}
        {/* =================================================== */}

        <section className="mt-8 grid grid-cols-1 gap-5 md:grid-cols-3">
          <PerformanceCard
            icon={ShieldIcon}
            title="Average Optimization Score"
            value={formatPercent(averageOverallOptimizationScore, 1)}
            description="Backend overall optimization score; not a health measurement"
            progress={averageOverallOptimizationScore ?? 0}
          />

          <PerformanceCard
            icon={Gauge}
            title="Average Utilization Fit Score"
            value={formatPercent(averageUtilizationFitScore, 1)}
            description="Rule-based fit between recorded utilization and resource sizing"
            progress={averageUtilizationFitScore ?? 0}
          />

          <PerformanceCard
            icon={Activity}
            title="Peak Utilization"
            value={Number.isFinite(maxUtilization) ? `${maxUtilization}%` : "N/A"}
            description="Highest available application-level utilization"
            progress={maxUtilization ?? 0}
          />
        </section>

        {/* =================================================== */}
        {/* RESOURCE UTILIZATION */}
        {/* =================================================== */}

        <section className="mt-8 rounded-3xl border border-slate-200 bg-white p-7 shadow-sm">
          <div>
            <div className="flex items-center gap-2">
              <Gauge
                size={22}
                className="text-blue-600"
              />

              <h2 className="text-xl font-bold text-slate-800">
                Cloud & Optimization Metrics
              </h2>
            </div>

            <p className="mt-2 text-sm text-slate-500">
              Application utilization, optimization decisions, cost-data coverage, and connected provider coverage. These are stored application metrics, not live infrastructure telemetry.
            </p>
          </div>

          <div className="mt-7 grid grid-cols-1 gap-5 md:grid-cols-2 xl:grid-cols-4">
            {resourceMetrics.map((resource) => {
              const ResourceIcon = resource.icon;

              return (
                <div
                  key={resource.name}
                  className="rounded-2xl border border-slate-100 bg-slate-50 p-5 transition hover:-translate-y-1 hover:shadow-md"
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <div className="rounded-xl bg-white p-3 text-blue-600 shadow-sm">
                        <ResourceIcon size={19} />
                      </div>

                      <div>
                        <p className="text-sm font-bold text-slate-700">
                          {resource.name}
                        </p>

                        <p className="text-[10px] text-slate-400">
                          {resource.description}
                        </p>
                      </div>
                    </div>

                    <span
                      className={`text-lg font-bold ${
                        Number.isFinite(resource.value)
                          ? getUtilizationColor(resource.progress ?? resource.value)
                          : "text-slate-400"
                      }`}
                    >
                      {resource.displayValue}
                    </span>
                  </div>

                  <div className="mt-5 h-2.5 overflow-hidden rounded-full bg-white">
                    <div
                      className={`h-full rounded-full transition-all duration-700 ${
                        Number.isFinite(resource.progress)
                          ? getUtilizationBar(resource.progress)
                          : "bg-slate-200"
                      }`}
                      style={{
                        width: `${Number.isFinite(resource.progress) ? Math.min(100, Math.max(0, resource.progress)) : 0}%`,
                      }}
                    />
                  </div>
                </div>
              );
            })}
          </div>
        </section>

        {/* =================================================== */}
        {/* PROVIDER + DECISION ANALYTICS */}
        {/* =================================================== */}

        <section className="mt-8 grid grid-cols-1 gap-6 xl:grid-cols-2">
          {/* PROVIDERS */}
          <div className="rounded-3xl border border-slate-200 bg-white p-7 shadow-sm">
            <div className="flex items-center gap-2">
              <Cloud
                size={22}
                className="text-blue-600"
              />

              <h2 className="text-xl font-bold text-slate-800">
                Cloud Provider Distribution
              </h2>
            </div>

            <p className="mt-2 text-sm text-slate-500">
              Infrastructure cost distribution across providers.
            </p>

            <div className="mt-7 space-y-5">
              {providerData.map((provider) => {
                const ProviderIcon = provider.icon;

                return (
                  <div
                    key={provider.name}
                    className="rounded-2xl border border-slate-100 bg-slate-50 p-5 transition hover:-translate-y-1 hover:shadow-md"
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-3">
                        <div className="rounded-xl bg-white p-3 text-blue-600 shadow-sm">
                          <ProviderIcon size={20} />
                        </div>

                        <div>
                          <p className="text-sm font-bold text-slate-700">
                            {provider.name}
                          </p>

                          <p className="text-xs text-slate-400">
                            {provider.services} services
                          </p>
                        </div>
                      </div>

                      <div className="text-right">
                        <p className="text-lg font-bold text-slate-800">
                          {formatMoney(provider.cost)}
                        </p>

                        <p className="text-xs font-semibold text-slate-400">
                          {provider.percentage}%
                        </p>
                      </div>
                    </div>

                    <p className="mt-3 text-xs text-slate-400">
                      {provider.description}
                    </p>

                    <div className="mt-4 h-2 overflow-hidden rounded-full bg-white">
                      <div
                        className="h-full rounded-full bg-gradient-to-r from-blue-500 to-violet-500 transition-all duration-700"
                        style={{
                          width: `${provider.percentage}%`,
                        }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* DECISIONS */}
          <div className="rounded-3xl border border-slate-200 bg-white p-7 shadow-sm">
            <div className="flex items-center gap-2">
              <BrainCircuit
                size={22}
                className="text-violet-600"
              />

              <h2 className="text-xl font-bold text-slate-800">
                AI Decision Distribution
              </h2>
            </div>

            <p className="mt-2 text-sm text-slate-500">
              Current optimization recommendations generated by
              CloudMorph.
            </p>

            <div className="mt-7 grid grid-cols-1 gap-3 sm:grid-cols-2">
              <button
                type="button"
                onClick={() => setSelectedDecision("ALL")}
                className={`rounded-xl border p-4 text-left transition ${
                  selectedDecision === "ALL"
                    ? "border-violet-300 bg-violet-50"
                    : "border-slate-200 bg-slate-50 hover:border-violet-200"
                }`}
              >
                <p className="text-xs font-bold text-slate-400">
                  ALL DECISIONS
                </p>

                <p className="mt-1 text-2xl font-bold text-slate-800">
                  {applications.length}
                </p>

                <p className="mt-1 text-[10px] text-slate-400">
                  Applications registered
                </p>
              </button>

              <div className="rounded-xl border border-emerald-100 bg-emerald-50 p-4">
                <p className="text-xs font-bold text-emerald-600">
                  SAVINGS OPPORTUNITY
                </p>

                <p className="mt-1 text-2xl font-bold text-emerald-700">
                  {formatMoney(potentialSavings, 0)}
                </p>

                <p className="mt-1 text-[10px] text-emerald-600">
                  Estimated monthly savings
                </p>
              </div>
            </div>

            <div className="mt-5 space-y-5">
              {decisionData.map((decision) => {
                const DecisionIcon = decision.icon;

                const isSelected =
                  selectedDecision === decision.label;

                return (
                  <button
                    key={decision.label}
                    type="button"
                    onClick={() =>
                      setSelectedDecision(
                        isSelected
                          ? "ALL"
                          : decision.label
                      )
                    }
                    className={`w-full rounded-2xl border p-5 text-left transition ${
                      isSelected
                        ? "border-violet-300 bg-violet-50 shadow-md"
                        : "border-slate-100 bg-slate-50 hover:border-violet-200 hover:bg-white"
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-3">
                        <div
                          className={`rounded-xl p-3 ${decision.lightStyle}`}
                        >
                          <DecisionIcon size={18} />
                        </div>

                        <div>
                          <div className="flex items-center gap-2">
                            <span className="text-sm font-bold text-slate-700">
                              {decision.label}
                            </span>

                            {isSelected && (
                              <CheckCircle2
                                size={15}
                                className="text-violet-500"
                              />
                            )}
                          </div>

                          <p className="mt-1 text-xs text-slate-400">
                            {decision.description}
                          </p>
                        </div>
                      </div>

                      <div className="text-right">
                        <p className="text-sm font-bold text-slate-700">
                          {decision.percentage}%
                        </p>

                        <p className="text-[10px] text-slate-400">
                          {decision.count} applications
                        </p>
                      </div>
                    </div>

                    <div className="mt-4 h-2 overflow-hidden rounded-full bg-white">
                      <div
                        className={`h-full rounded-full transition-all duration-700 ${decision.style}`}
                        style={{
                          width: `${decision.percentage}%`,
                        }}
                      />
                    </div>
                  </button>
                );
              })}
            </div>

            <div className="mt-8 rounded-2xl border border-violet-100 bg-gradient-to-r from-violet-50 to-blue-50 p-5">
              <div className="flex gap-3">
                <div className="rounded-xl bg-white p-3 text-violet-600 shadow-sm">
                  <Sparkles size={19} />
                </div>

                <div>
                  <p className="text-sm font-bold text-slate-700">
                    AI Optimization Insight
                  </p>

                  <p className="mt-1 text-xs leading-relaxed text-slate-500">
                    Decision counts reflect the latest decision stored on each
                    application. Detailed analysis history is shown separately
                    in the backend recommendation records.
                  </p>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* =================================================== */}
        {/* APPLICATION ANALYTICS */}
        {/* =================================================== */}

        <section className="mt-8">
          <div className="flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between">
            <div>
              <div className="flex items-center gap-2">
                <Server
                  size={22}
                  className="text-blue-600"
                />

                <h2 className="text-xl font-bold text-slate-800">
                  Application Analytics
                </h2>
              </div>

              <p className="mt-2 text-sm text-slate-500">
                Cost, utilization, and optimization opportunities
                by application.
              </p>
            </div>

            <div className="rounded-xl bg-white px-4 py-2 text-xs font-semibold text-slate-500 shadow-sm">
              Showing{" "}
              <span className="font-bold text-slate-800">
                {visibleApplications.length}
              </span>{" "}
              applications
            </div>
          </div>

          <div className="mt-6 grid grid-cols-1 gap-5 xl:grid-cols-2">
            {visibleApplications.map((application) => (
              <div
                key={application.id}
                className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm transition hover:-translate-y-1 hover:shadow-lg"
              >
                <div className="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
                  <div className="flex items-start gap-4">
                    <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-gradient-to-br from-blue-600 to-violet-600 text-white shadow-md">
                      <Layers3 size={21} />
                    </div>

                    <div>
                      <h3 className="text-lg font-bold text-slate-800">
                        {application.name}
                      </h3>

                      <p className="mt-1 text-xs text-slate-400">
                        {application.category}
                      </p>

                      <div className="mt-2 flex flex-wrap gap-2">
                        <span className="rounded-lg bg-slate-100 px-2.5 py-1 text-[10px] font-bold text-slate-600">
                          {application.provider}
                        </span>

                        <span
                          className={`rounded-lg border px-2.5 py-1 text-[10px] font-bold ${getDecisionStyle(
                            application.decision
                          )}`}
                        >
                          {application.decision}
                        </span>

                        <span
                          className={`rounded-lg border px-2.5 py-1 text-[10px] font-bold ${getStatusStyle(
                            application.status
                          )}`}
                        >
                          {application.status}
                        </span>
                      </div>
                    </div>
                  </div>

                  <div className="text-left sm:text-right">
                    <p className="text-xs text-slate-400">
                      Monthly Cost
                    </p>

                    <p className="mt-1 text-xl font-bold text-slate-800">
                      {formatMoney(application.cost)}
                    </p>

                    <p className="mt-1 flex items-center gap-1 text-xs font-semibold text-emerald-600 sm:justify-end">
                      {Number.isFinite(application.savings) ? (
                        <>
                          <TrendingDown size={13} />
                          {formatMoney(application.savings)} estimated
                        </>
                      ) : (
                        <span className="text-slate-400">Savings: N/A</span>
                      )}
                    </p>
                  </div>
                </div>

                <div className="mt-6 grid grid-cols-2 gap-3 sm:grid-cols-4">
                  <MiniMetric
                    label="Optimized"
                    value={`${formatMoney(application.optimizedCost)}`}
                  />

                  <MiniMetric
                    label="Estimated Savings"
                    value={`${formatMoney(application.savings)}`}
                  />

                  <MiniMetric
                    label="Utilization"
                    value={Number.isFinite(application.utilization) ? `${application.utilization}%` : "N/A"}
                  />

                  <MiniMetric
                    label="Services"
                    value={application.services}
                  />
                </div>

                <div className="mt-6">
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-bold text-slate-600">
                      Resource Utilization
                    </span>

                    <span
                      className={`text-xs font-bold ${
                        Number.isFinite(application.utilization)
                          ? getUtilizationColor(application.utilization)
                          : "text-slate-400"
                      }`}
                    >
                      {Number.isFinite(application.utilization)
                        ? `${application.utilization}%`
                        : "N/A"}
                    </span>
                  </div>

                  <div className="mt-2 h-2 overflow-hidden rounded-full bg-slate-100">
                    <div
                      className={`h-full rounded-full transition-all duration-700 ${
                        Number.isFinite(application.utilization)
                          ? getUtilizationBar(application.utilization)
                          : "bg-slate-200"
                      }`}
                      style={{
                        width: `${Number.isFinite(application.utilization) ? application.utilization : 0}%`,
                      }}
                    />
                  </div>
                </div>

                {/* ACTIONS */}
                <div className="mt-6 flex flex-col gap-3 sm:flex-row">
                  <button
                    type="button"
                    onClick={() =>
                      setSelectedApplication(application)
                    }
                    className="flex flex-1 items-center justify-center gap-2 rounded-xl border border-slate-200 bg-white px-4 py-3 text-xs font-bold text-slate-600 transition hover:border-blue-200 hover:bg-blue-50 hover:text-blue-600"
                  >
                    <BarChart3 size={15} />
                    Quick View
                  </button>

                  <button
                    type="button"
                    onClick={() =>
                      handleViewDetails(application)
                    }
                    className="flex flex-1 items-center justify-center gap-2 rounded-xl bg-slate-900 px-4 py-3 text-xs font-bold text-white transition hover:bg-slate-800"
                  >
                    <ArrowUpRight size={15} />
                    View Details
                  </button>

                  <button
                    type="button"
                    onClick={() =>
                      handleViewAnalysis(application)
                    }
                    className="flex flex-1 items-center justify-center gap-2 rounded-xl bg-gradient-to-r from-violet-600 to-blue-600 px-4 py-3 text-xs font-bold text-white transition hover:from-violet-700 hover:to-blue-700"
                  >
                    <BrainCircuit size={15} />
                    AI Analysis
                  </button>
                </div>
              </div>
            ))}
          </div>

          {visibleApplications.length === 0 && (
            <div className="mt-6 rounded-3xl border border-dashed border-slate-300 bg-white p-12 text-center">
              <div className="mx-auto flex h-14 w-14 items-center justify-center rounded-2xl bg-slate-100 text-slate-400">
                <Server size={25} />
              </div>

              <h3 className="mt-4 text-lg font-bold text-slate-700">
                No applications found
              </h3>

              <p className="mt-2 text-sm text-slate-400">
                No applications currently match the selected
                decision.
              </p>

              <button
                type="button"
                onClick={() => setSelectedDecision("ALL")}
                className="mt-5 rounded-xl bg-slate-900 px-5 py-3 text-xs font-bold text-white"
              >
                Show All Applications
              </button>
            </div>
          )}
        </section>

        {/* =================================================== */}
        {/* KEY INSIGHTS */}
        {/* =================================================== */}

        <section className="mt-8 grid grid-cols-1 gap-5 md:grid-cols-3">
          <InsightCard
            icon={DollarSign}
            title="Highest Cost"
            value={highestCostApplication?.name || "No applications"}
            description={highestCostApplication ? `${formatMoney(highestCostApplication.cost)} monthly infrastructure cost` : "Add an application to see cost insights"}
            iconStyle="bg-blue-50 text-blue-600"
          />

          <InsightCard
            icon={TrendingDown}
            title="Highest Savings"
            value={highestSavingsApplication?.name || "No applications"}
            description={highestSavingsApplication ? `${formatMoney(highestSavingsApplication.savings)} estimated monthly savings from a matching saved scenario` : "No eligible saved RESIZE/SWITCH savings scenario is available"}
            iconStyle="bg-emerald-50 text-emerald-600"
          />

          <InsightCard
            icon={Gauge}
            title="Peak Utilization"
            value={Number.isFinite(maxUtilization) ? `${maxUtilization}%` : "N/A"}
            description="Highest available application-level utilization"
            iconStyle="bg-violet-50 text-violet-600"
          />
        </section>

        {/* =================================================== */}
        {/* ANALYTICS SUMMARY */}
        {/* =================================================== */}

        <section className="mt-10 overflow-hidden rounded-3xl border border-blue-100 bg-gradient-to-r from-blue-50 via-white to-violet-50 p-8">
          <div className="flex flex-col gap-7 lg:flex-row lg:items-center lg:justify-between">
            <div className="flex items-start gap-5">
              <div className="rounded-2xl bg-gradient-to-br from-blue-600 to-violet-600 p-4 text-white shadow-lg">
                <BarChart3 size={28} />
              </div>

              <div>
                <p className="text-xs font-bold uppercase tracking-[0.2em] text-violet-600">
                  CloudMorph Analytics Intelligence
                </p>

                <h2 className="mt-3 text-2xl font-bold text-slate-800">
                  Application costs and utilization are summarized.
                </h2>

                <p className="mt-3 max-w-2xl text-sm leading-relaxed text-slate-500">
                  Analytics summarizes stored application records and comparable
                  cost estimates. Savings are estimates, not verified billing
                  reductions. Cloud provider billing and live resource telemetry
                  are not connected yet.
                </p>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-3">
              <SummaryMetric
                label="Cost"
                value={formatMoney(currentCost, 0)}
              />

              <SummaryMetric
                label="Estimated Savings"
                value={`${formatMoney(potentialSavings, 0)}`}
              />

              <SummaryMetric
                label="Utilization"
                value={Number.isFinite(averageUtilization) ? `${averageUtilization.toFixed(1)}%` : "N/A"}
              />

              <SummaryMetric
                label="Apps"
                value={applications.length}
              />
            </div>
          </div>
        </section>

        <div className="h-6" />
      </main>

      {/* ===================================================== */}
      {/* APPLICATION QUICK VIEW MODAL */}
      {/* ===================================================== */}

      {selectedApplication && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/60 p-4 backdrop-blur-sm"
          onClick={() => setSelectedApplication(null)}
        >
          <div
            className="max-h-[90vh] w-full max-w-2xl overflow-y-auto rounded-3xl bg-white shadow-2xl"
            onClick={(event) => event.stopPropagation()}
          >
            <div className="bg-gradient-to-r from-slate-950 via-blue-950 to-violet-950 p-7 text-white">
              <div className="flex items-start justify-between gap-4">
                <div className="flex items-start gap-4">
                  <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-white/10">
                    <Layers3 size={22} />
                  </div>

                  <div>
                    <p className="text-xs font-bold uppercase tracking-wider text-blue-200">
                      Application Analytics
                    </p>

                    <h2 className="mt-2 text-2xl font-bold">
                      {selectedApplication.name}
                    </h2>

                    <p className="mt-1 text-sm text-slate-300">
                      {selectedApplication.category} ·{" "}
                      {selectedApplication.provider}
                    </p>
                  </div>
                </div>

                <button
                  type="button"
                  onClick={() =>
                    setSelectedApplication(null)
                  }
                  className="rounded-xl bg-white/10 p-2 text-slate-300 transition hover:bg-white/20 hover:text-white"
                >
                  <X size={19} />
                </button>
              </div>
            </div>

            <div className="p-7">
              <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
                <ModalMetric
                  label="Monthly Cost"
                  value={`${formatMoney(selectedApplication.cost)}`}
                />

                <ModalMetric
                  label="Optimized"
                  value={`${formatMoney(selectedApplication.optimizedCost)}`}
                />

                <ModalMetric
                  label="Estimated Savings"
                  value={`${formatMoney(selectedApplication.savings)}`}
                />

                <ModalMetric
                  label="Services"
                  value={selectedApplication.services}
                />
              </div>

              <div className="mt-6 grid grid-cols-1 gap-4 md:grid-cols-2">
                <DetailMetric
                  label="Resource Utilization"
                  value={Number.isFinite(selectedApplication.utilization) ? `${selectedApplication.utilization}%` : "N/A"}
                  progress={selectedApplication.utilization ?? 0}
                  progressClass={Number.isFinite(selectedApplication.utilization) ? getUtilizationBar(selectedApplication.utilization) : "bg-slate-200"}
                />

                <DetailMetric
                  label="Overall Optimization Score"
                  value={Number.isFinite(selectedApplication.overallOptimizationScore) ? `${selectedApplication.overallOptimizationScore}%` : "N/A"}
                  progress={selectedApplication.overallOptimizationScore ?? 0}
                  progressClass="bg-emerald-500"
                />

                <DetailMetric
                  label="Utilization Fit Score"
                  value={Number.isFinite(selectedApplication.optimizationScore) ? `${selectedApplication.optimizationScore}%` : "N/A"}
                  progress={selectedApplication.optimizationScore ?? 0}
                  progressClass="bg-violet-500"
                />

                <DetailMetric
                  label="Optimization Decision"
                  value={selectedApplication.decision || "N/A"}
                  progress={0}
                  progressClass="bg-blue-500"
                />
              </div>

              <div className="mt-6 rounded-2xl border border-violet-100 bg-gradient-to-r from-violet-50 to-blue-50 p-5">
                <div className="flex items-start gap-3">
                  <div className="rounded-xl bg-white p-3 text-violet-600 shadow-sm">
                    <BrainCircuit size={19} />
                  </div>

                  <div>
                    <p className="text-sm font-bold text-slate-700">
                      CloudMorph Decision
                    </p>

                    <div className="mt-2 flex items-center gap-2">
                      <span
                        className={`rounded-lg border px-3 py-1.5 text-xs font-bold ${getDecisionStyle(
                          selectedApplication.decision
                        )}`}
                      >
                        {selectedApplication.decision}
                      </span>

                      <span className="text-xs text-slate-500">
                        Recommended optimization path
                      </span>
                    </div>

                    <p className="mt-3 text-xs leading-relaxed text-slate-500">
                      This is the decision currently stored for this application. Review its
                      recommendation details before making infrastructure changes.
                    </p>
                  </div>
                </div>
              </div>

              <div className="mt-6 flex flex-col gap-3 sm:flex-row">
                <button
                  type="button"
                  onClick={() => {
                    setSelectedApplication(null);
                    handleViewDetails(selectedApplication);
                  }}
                  className="flex flex-1 items-center justify-center gap-2 rounded-xl bg-slate-900 px-5 py-3 text-sm font-bold text-white transition hover:bg-slate-800"
                >
                  <ArrowUpRight size={17} />
                  View Application
                </button>

                <button
                  type="button"
                  onClick={() => {
                    setSelectedApplication(null);
                    handleViewAnalysis(selectedApplication);
                  }}
                  className="flex flex-1 items-center justify-center gap-2 rounded-xl bg-gradient-to-r from-violet-600 to-blue-600 px-5 py-3 text-sm font-bold text-white transition hover:from-violet-700 hover:to-blue-700"
                >
                  <BrainCircuit size={17} />
                  Run AI Analysis
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

/* ========================================================= */
/* HERO METRIC */
/* ========================================================= */

function HeroMetric({
  icon: Icon,
  label,
  value,
  description,
}) {
  return (
    <div className="rounded-2xl border border-white/10 bg-white/5 p-5 backdrop-blur transition hover:bg-white/10">
      <div className="flex items-center gap-3">
        <div className="rounded-xl bg-white/10 p-3 text-blue-200">
          <Icon size={21} />
        </div>

        <div>
          <p className="text-xs text-slate-400">
            {label}
          </p>

          <p className="mt-1 text-2xl font-bold text-white">
            {value}
          </p>
        </div>
      </div>

      <p className="mt-4 text-xs text-slate-500">
        {description}
      </p>
    </div>
  );
}

/* ========================================================= */
/* PERFORMANCE CARD */
/* ========================================================= */

function PerformanceCard({
  icon: Icon,
  title,
  value,
  description,
  progress,
}) {
  return (
    <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm transition hover:-translate-y-1 hover:shadow-lg">
      <div className="flex items-center justify-between">
        <div className="rounded-xl bg-blue-50 p-3 text-blue-600">
          <Icon size={20} />
        </div>

        <span className="text-xl font-bold text-slate-800">
          {value}
        </span>
      </div>

      <p className="mt-5 text-sm font-bold text-slate-700">
        {title}
      </p>

      <p className="mt-1 text-xs text-slate-400">
        {description}
      </p>

      {Number.isFinite(progress) && progress > 0 && (
        <div className="mt-4 h-2 overflow-hidden rounded-full bg-slate-100">
          <div
            className="h-full rounded-full bg-blue-500 transition-all duration-700"
            style={{
              width: `${Math.min(100, Math.max(0, progress))}%`,
            }}
          />
        </div>
      )}
    </div>
  );
}

/* ========================================================= */
/* IMPACT ROW */
/* ========================================================= */

function ImpactRow({
  label,
  value,
  icon: Icon,
  highlight = false,
}) {
  return (
    <div
      className={`flex items-center justify-between rounded-xl p-4 ${
        highlight
          ? "bg-emerald-50"
          : "bg-slate-50"
      }`}
    >
      <div className="flex items-center gap-3">
        <div
          className={`rounded-lg bg-white p-2 ${
            highlight
              ? "text-emerald-600"
              : "text-slate-500"
          } shadow-sm`}
        >
          <Icon size={16} />
        </div>

        <span className="text-xs font-bold text-slate-600">
          {label}
        </span>
      </div>

      <span
        className={`text-sm font-bold ${
          highlight
            ? "text-emerald-600"
            : "text-slate-800"
        }`}
      >
        {value}
      </span>
    </div>
  );
}

/* ========================================================= */
/* MINI METRIC */
/* ========================================================= */

function MiniMetric({
  label,
  value,
}) {
  return (
    <div className="rounded-xl bg-slate-50 p-3">
      <p className="text-[10px] font-medium text-slate-400">
        {label}
      </p>

      <p className="mt-1 text-sm font-bold text-slate-700">
        {value}
      </p>
    </div>
  );
}

/* ========================================================= */
/* INSIGHT CARD */
/* ========================================================= */

function InsightCard({
  icon: Icon,
  title,
  value,
  description,
  iconStyle,
}) {
  return (
    <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm transition hover:-translate-y-1 hover:shadow-lg">
      <div className="flex items-start justify-between">
        <div
          className={`rounded-xl p-3 ${iconStyle}`}
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

      <p className="mt-2 text-lg font-bold text-slate-800">
        {value}
      </p>

      <p className="mt-1 text-xs leading-relaxed text-slate-400">
        {description}
      </p>
    </div>
  );
}

/* ========================================================= */
/* SUMMARY METRIC */
/* ========================================================= */

function SummaryMetric({
  label,
  value,
}) {
  return (
    <div className="rounded-xl bg-white p-4 shadow-sm">
      <p className="text-[10px] font-medium text-slate-400">
        {label}
      </p>

      <p className="mt-1 text-lg font-bold text-slate-800">
        {value}
      </p>
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
      <p className="text-[10px] font-medium text-slate-400">
        {label}
      </p>

      <p className="mt-1 text-lg font-bold text-slate-800">
        {value}
      </p>
    </div>
  );
}

/* ========================================================= */
/* DETAIL METRIC */
/* ========================================================= */

function DetailMetric({
  label,
  value,
  progress,
  progressClass,
}) {
  return (
    <div className="rounded-2xl border border-slate-100 bg-slate-50 p-4">
      <div className="flex items-center justify-between">
        <p className="text-xs font-bold text-slate-600">
          {label}
        </p>

        <p className="text-sm font-bold text-slate-800">
          {value}
        </p>
      </div>

      {Number.isFinite(progress) && progress > 0 && (
        <div className="mt-3 h-2 overflow-hidden rounded-full bg-white">
          <div
            className={`h-full rounded-full transition-all duration-700 ${progressClass}`}
            style={{
              width: `${Math.min(100, Math.max(0, progress))}%`,
            }}
          />
        </div>
      )}
    </div>
  );
}

/* ========================================================= */
/* SHIELD ICON */
/* ========================================================= */

function ShieldIcon({ size = 20 }) {
  return (
    <div className="flex items-center justify-center">
      <svg
        width={size}
        height={size}
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      >
        <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10Z" />
        <path d="m9 12 2 2 4-4" />
      </svg>
    </div>
  );
}

export default Analytics;