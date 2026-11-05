import { useCallback, useEffect, useMemo, useState } from "react";
import API from "../services/api";

import {
  Activity,
  AlertTriangle,
  ArrowDownRight,
  ArrowUpRight,
  BarChart3,
  BrainCircuit,
  Check,
  CheckCircle2,
  ChevronDown,
  Cloud,
  Cpu,
  Database,
  DollarSign,
  Filter,
  HardDrive,
  Layers3,
  Network,
  RefreshCw,
  Search,
  Server,
  ShieldCheck,
  SlidersHorizontal,
  Sparkles,
  TrendingDown,
  X,
  Zap,
} from "lucide-react";

function Infrastructure() {
  /* ========================================================= */
  /* STATE */
  /* ========================================================= */

  const [searchQuery, setSearchQuery] = useState("");
  const [providerFilter, setProviderFilter] = useState("All");
  const [statusFilter, setStatusFilter] = useState("All");
  const [sortBy, setSortBy] = useState("utilization");
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [loadError, setLoadError] = useState("");
  const [services, setServices] = useState([]);
  const [selectedResource, setSelectedResource] = useState(null);

  /* ========================================================= */
  /* LOAD APPLICATIONS FROM THE BACKEND */
  /* ========================================================= */

  const normalizeService = (app, index) => {
    const toFiniteNumber = (value, fallback = 0) => {
      if (value === null || value === undefined || value === "") return fallback;
      const number = Number(value);
      return Number.isFinite(number) ? number : fallback;
    };

    const monthlyCost = Math.max(
      0,
      toFiniteNumber(app.monthly_cost ?? app.monthlyCost ?? app.cost)
    );
    const rawOptimizedCost = toFiniteNumber(
      app.optimized_cost ?? app.optimizedCost ?? app.optimized_cost_monthly,
      monthlyCost
    );
    const optimizedCost = Math.min(monthlyCost, Math.max(0, rawOptimizedCost));
    const toOptionalPercent = (value) => {
      if (value === null || value === undefined || value === "") return null;
      const number = toFiniteNumber(value, NaN);
      return Number.isFinite(number) && number >= 0 && number <= 100
        ? number
        : null;
    };

    const utilization = toOptionalPercent(
      app.utilization ?? app.usagePercent ?? app.utilization_percent
    );
    const decision = String(app.decision ?? app.optimization ?? "KEEP").toUpperCase();
    const provider = String(app.provider ?? "Not configured").trim().toUpperCase() || "NOT CONFIGURED";
    const serviceType = String(app.service ?? app.category ?? app.type ?? "Application");
    const statusValue = String(app.status ?? "").toLowerCase();
    const status =
      statusValue === "warning" || statusValue === "error" || (utilization !== null && utilization >= 85)
        ? "Warning"
        : "Healthy";

    let Icon = Layers3;
    if (/compute|ec2|server|vm/i.test(serviceType)) Icon = Cpu;
    else if (/database|sql|mongo|postgres|rds/i.test(serviceType)) Icon = Database;
    else if (/storage|s3|bucket/i.test(serviceType)) Icon = HardDrive;
    else if (/api|network|service/i.test(serviceType)) Icon = Network;

    return {
      id: String(app.id ?? app._id ?? `application-${index}`),
      name: String(app.service ?? app.name ?? `Application ${index + 1}`),
      application: String(app.name ?? "Unnamed application"),
      provider,
      region: app.region ? String(app.region) : "Not configured",
      type: serviceType,
      icon: Icon,
      instance: app.instance ?? app.configuration ?? null,
      status,
      utilization,
      cpu: toOptionalPercent(app.cpu_utilization ?? app.cpu),
      memory: toOptionalPercent(app.memory_utilization ?? app.memory),
      storage: toOptionalPercent(app.storage_utilization ?? app.storage),
      network: toOptionalPercent(app.network_utilization ?? app.network),
      monthlyCost,
      optimization: ["KEEP", "RESIZE", "SWITCH"].includes(decision) ? decision : "KEEP",
      potentialSavings: Math.max(0, monthlyCost - optimizedCost),
      uptime: toOptionalPercent(app.uptime_percent ?? app.uptime),
      recommendation:
        app.recommendation ??
        (decision === "RESIZE"
          ? "Review resource sizing against the application's recorded utilization."
          : decision === "SWITCH"
          ? "Compare compatible alternatives before making any infrastructure change."
          : "No optimization change is recorded for this application."),
    };
  };

  const fetchInfrastructure = useCallback(async () => {
    setLoadError("");
    try {
      const response = await API.get("/applications");
      const payload = response.data;
      const records = Array.isArray(payload)
        ? payload
        : Array.isArray(payload?.applications)
        ? payload.applications
        : Array.isArray(payload?.data)
        ? payload.data
        : payload && typeof payload === "object" && Object.keys(payload).length === 0
        ? []
        : null;

      if (!records) {
        throw new Error("The backend returned an unexpected applications response.");
      }

      setServices(records
        .filter((app) => app && typeof app === "object" && !Array.isArray(app))
        .map(normalizeService));
    } catch (error) {
      setLoadError(
        error.response?.data?.detail ||
          error.message ||
          "Unable to load applications from the backend."
      );
      setServices([]);
    } finally {
      setIsLoading(false);
      setIsRefreshing(false);
    }
  }, []);

  useEffect(() => {
    fetchInfrastructure();
  }, [fetchInfrastructure]);

  /* ========================================================= */
  /* CALCULATIONS */
  /* ========================================================= */

  const totalMonthlyCost = services.reduce(
    (total, service) => total + service.monthlyCost,
    0
  );

  const totalPotentialSavings = services.reduce(
    (total, service) => total + service.potentialSavings,
    0
  );

  const servicesWithUtilization = services.filter(
    (service) => service.utilization !== null && Number.isFinite(service.utilization)
  );

  const averageUtilization = servicesWithUtilization.length
    ? Math.round(
        servicesWithUtilization.reduce((total, service) => total + service.utilization, 0) /
          servicesWithUtilization.length
      )
    : null;

  const highestUtilizationService = servicesWithUtilization.length
    ? servicesWithUtilization.reduce((highest, service) =>
        service.utilization > highest.utilization ? service : highest
      )
    : null;

  const lowestUtilizationService = servicesWithUtilization.length
    ? servicesWithUtilization.reduce((lowest, service) =>
        service.utilization < lowest.utilization ? service : lowest
      )
    : null;

  const utilizationAvailability = services.length
    ? Math.round((servicesWithUtilization.length / services.length) * 100)
    : null;

  const healthyServices = services.filter((service) => service.status === "Healthy").length;
  const warningServices = services.filter((service) => service.status === "Warning").length;
  const resizeServices = services.filter((service) => service.optimization === "RESIZE").length;
  const switchServices = services.filter((service) => service.optimization === "SWITCH").length;

  /* ========================================================= */
  /* FILTERING */
  /* ========================================================= */

  const filteredServices = useMemo(() => {
    let result = services.filter((service) => {
      const query = searchQuery.toLowerCase();

      const matchesSearch =
        service.name.toLowerCase().includes(query) ||
        service.application.toLowerCase().includes(query) ||
        service.type.toLowerCase().includes(query) ||
        service.provider.toLowerCase().includes(query);

      const matchesProvider =
        providerFilter === "All" ||
        service.provider === providerFilter;

      const matchesStatus =
        statusFilter === "All" ||
        service.status === statusFilter;

      return (
        matchesSearch &&
        matchesProvider &&
        matchesStatus
      );
    });

    result.sort((a, b) => {
      if (sortBy === "utilization") {
        return (b.utilization ?? -1) - (a.utilization ?? -1);
      }

      if (sortBy === "cost") {
        return b.monthlyCost - a.monthlyCost;
      }

      if (sortBy === "savings") {
        return b.potentialSavings - a.potentialSavings;
      }

      if (sortBy === "uptime") {
        // Keep valid zero uptime values; place unavailable values last.
        if (a.uptime === null && b.uptime === null) return 0;
        if (a.uptime === null) return 1;
        if (b.uptime === null) return -1;
        return b.uptime - a.uptime;
      }

      return 0;
    });

    return result;
  }, [
    searchQuery,
    providerFilter,
    statusFilter,
    sortBy,
    services,
  ]);

  /* ========================================================= */
  /* REFRESH */
  /* ========================================================= */

  const handleRefresh = () => {
    setIsRefreshing(true);
    fetchInfrastructure();
  };

  /* ========================================================= */
  /* RESOURCE MODAL */
  /* ========================================================= */

  const handleViewResource = (service) => {
    setSelectedResource(service);
  };

  const handleCloseResource = () => {
    setSelectedResource(null);
  };

  /* ========================================================= */
  /* HELPERS */
  /* ========================================================= */

  const getStatusStyle = (status) => {
    if (status === "Healthy") {
      return "bg-emerald-50 text-emerald-600 border-emerald-100";
    }

    return "bg-amber-50 text-amber-600 border-amber-100";
  };

  const getOptimizationStyle = (optimization) => {
    if (optimization === "KEEP") {
      return "bg-emerald-50 text-emerald-600 border-emerald-100";
    }

    if (optimization === "RESIZE") {
      return "bg-blue-50 text-blue-600 border-blue-100";
    }

    return "bg-violet-50 text-violet-600 border-violet-100";
  };

  const getUtilizationStyle = (value) => {
    if (value === null || value === undefined) {
      return {
        text: "text-slate-400",
        bar: "bg-slate-200",
      };
    }

    if (value >= 85) {
      return {
        text: "text-red-500",
        bar: "bg-red-500",
      };
    }

    if (value >= 70) {
      return {
        text: "text-amber-500",
        bar: "bg-amber-500",
      };
    }

    return {
      text: "text-emerald-500",
      bar: "bg-emerald-500",
    };
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
                <span className="flex items-center gap-2 rounded-full border border-blue-400/20 bg-blue-500/10 px-4 py-2 text-xs font-bold uppercase tracking-wider text-blue-200">
                  <Cloud size={15} />
                  Cloud Infrastructure
                </span>

                <span className="flex items-center gap-2 text-xs font-semibold text-emerald-300">
                  <span className="h-2 w-2 animate-pulse rounded-full bg-emerald-400" />
                  Application Data Connected
                </span>
              </div>

              <h1 className="mt-6 text-3xl font-bold tracking-tight text-white md:text-5xl">
                Infrastructure
              </h1>

              <p className="mt-4 max-w-2xl text-sm leading-relaxed text-slate-300 md:text-base">
                Monitor cloud resources, utilization, costs, health,
                and optimization opportunities across your connected
                infrastructure.
              </p>

              <div className="mt-5 flex flex-wrap items-center gap-4 text-xs text-slate-400">
                <span className="flex items-center gap-2">
                  <Activity size={15} />
                  Saved resource metrics
                </span>

                <span className="flex items-center gap-2">
                  <ShieldCheck size={15} />
                  {healthyServices} healthy services
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
                  isRefreshing ? "animate-spin" : ""
                }
              />

              {isRefreshing
                ? "Refreshing..."
                : "Refresh Infrastructure"}
            </button>
          </div>

          {/* HERO METRICS */}
          <div className="mt-10 grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-4">
            <HeroMetric
              icon={Layers3}
              label="Cloud Services"
              value={services.length}
              description="Resources monitored"
            />

            <HeroMetric
              icon={DollarSign}
              label="Monthly Cost"
              value={`$${totalMonthlyCost.toFixed(2)}`}
              description="Current infrastructure"
            />

            <HeroMetric
              icon={TrendingDown}
              label="Potential Savings"
              value={`$${totalPotentialSavings.toFixed(2)}`}
              description="Estimated monthly"
            />

            <HeroMetric
              icon={Activity}
              label="Avg Utilization"
              value={averageUtilization === null ? "—" : `${averageUtilization}%`}
              description="Across monitored services"
            />
          </div>
        </div>
      </section>

      {/* ===================================================== */}
      {/* MAIN */}
      {/* ===================================================== */}

      <main className="p-8">
        {/* =================================================== */}
        {/* OVERVIEW CARDS */}
        {/* =================================================== */}

        <div className="grid grid-cols-1 gap-5 md:grid-cols-2 xl:grid-cols-4">
          <OverviewCard
            icon={CheckCircle2}
            title="Healthy Services"
            value={healthyServices}
            description={`of ${services.length} monitored services`}
            iconStyle="bg-emerald-50 text-emerald-600"
          />

          <OverviewCard
            icon={AlertTriangle}
            title="Attention Required"
            value={warningServices}
            description="Services with warnings"
            iconStyle="bg-amber-50 text-amber-600"
          />

          <OverviewCard
            icon={SlidersHorizontal}
            title="Resize Opportunities"
            value={resizeServices}
            description="Resource optimization candidates"
            iconStyle="bg-blue-50 text-blue-600"
          />

          <OverviewCard
            icon={Sparkles}
            title="Switch Opportunities"
            value={switchServices}
            description="Alternative service candidates"
            iconStyle="bg-violet-50 text-violet-600"
          />
        </div>

        {/* =================================================== */}
        {/* UTILIZATION OVERVIEW */}
        {/* =================================================== */}

        <section className="mt-8 rounded-3xl border border-slate-200 bg-white p-7 shadow-sm">
          <div className="flex flex-col gap-2">
            <div className="flex items-center gap-2">
              <BarChart3
                size={22}
                className="text-blue-600"
              />

              <h2 className="text-xl font-bold text-slate-800">
                Resource Utilization
              </h2>
            </div>

            <p className="text-sm text-slate-500">
              Current utilization levels across your cloud
              infrastructure.
            </p>
          </div>

          <div className="mt-7 grid grid-cols-1 gap-5 md:grid-cols-2 xl:grid-cols-4">
            <UtilizationCard
              icon={Activity}
              label="Average Utilization"
              value={averageUtilization}
              description={`${servicesWithUtilization.length} of ${services.length} applications reporting`}
            />

            <UtilizationCard
              icon={ArrowUpRight}
              label="Highest Utilization"
              value={highestUtilizationService?.utilization ?? null}
              description={
                highestUtilizationService
                  ? highestUtilizationService.application
                  : "No utilization data available"
              }
            />

            <UtilizationCard
              icon={ArrowDownRight}
              label="Lowest Utilization"
              value={lowestUtilizationService?.utilization ?? null}
              description={
                lowestUtilizationService
                  ? lowestUtilizationService.application
                  : "No utilization data available"
              }
            />

            <UtilizationCard
              icon={Database}
              label="Data Availability"
              value={utilizationAvailability}
              description={`${servicesWithUtilization.length} of ${services.length} applications reporting`}
            />
          </div>
        </section>

        {/* =================================================== */}
        {/* FILTERS */}
        {/* =================================================== */}

        <section className="mt-8">
          <div className="flex flex-col gap-5 xl:flex-row xl:items-end xl:justify-between">
            <div>
              <div className="flex items-center gap-2">
                <Server
                  size={22}
                  className="text-violet-600"
                />

                <h2 className="text-xl font-bold text-slate-800">
                  Cloud Services
                </h2>
              </div>

              <p className="mt-2 text-sm text-slate-500">
                Monitor individual services and their optimization
                status.
              </p>
            </div>

            <div className="flex flex-col gap-3 md:flex-row">
              {/* SEARCH */}
              <div className="relative">
                <Search
                  size={18}
                  className="absolute left-4 top-1/2 -translate-y-1/2 text-slate-400"
                />

                <input
                  type="text"
                  value={searchQuery}
                  onChange={(event) =>
                    setSearchQuery(event.target.value)
                  }
                  placeholder="Search services..."
                  className="w-full rounded-xl border border-slate-200 bg-white py-3 pl-11 pr-4 text-sm outline-none transition focus:border-blue-400 focus:ring-4 focus:ring-blue-100 md:w-56"
                />
              </div>

              {/* PROVIDER */}
              <div className="relative">
                <Cloud
                  size={17}
                  className="pointer-events-none absolute left-4 top-1/2 -translate-y-1/2 text-slate-400"
                />

                <select
                  value={providerFilter}
                  onChange={(event) =>
                    setProviderFilter(event.target.value)
                  }
                  className="appearance-none rounded-xl border border-slate-200 bg-white py-3 pl-11 pr-10 text-sm font-medium text-slate-600 outline-none focus:border-blue-400 focus:ring-4 focus:ring-blue-100"
                >
                  <option value="All">
                    All Providers
                  </option>

                  {Array.from(new Set(services.map((service) => service.provider)))
                    .filter((provider) => provider && provider !== "NOT CONFIGURED")
                    .map((provider) => (
                      <option key={provider} value={provider}>{provider}</option>
                    ))}
                </select>

                <ChevronDown
                  size={16}
                  className="pointer-events-none absolute right-4 top-1/2 -translate-y-1/2 text-slate-400"
                />
              </div>

              {/* STATUS */}
              <div className="relative">
                <Filter
                  size={17}
                  className="pointer-events-none absolute left-4 top-1/2 -translate-y-1/2 text-slate-400"
                />

                <select
                  value={statusFilter}
                  onChange={(event) =>
                    setStatusFilter(event.target.value)
                  }
                  className="appearance-none rounded-xl border border-slate-200 bg-white py-3 pl-11 pr-10 text-sm font-medium text-slate-600 outline-none focus:border-blue-400 focus:ring-4 focus:ring-blue-100"
                >
                  <option value="All">All Status</option>

                  <option value="Healthy">
                    Healthy
                  </option>

                  <option value="Warning">
                    Warning
                  </option>
                </select>

                <ChevronDown
                  size={16}
                  className="pointer-events-none absolute right-4 top-1/2 -translate-y-1/2 text-slate-400"
                />
              </div>

              {/* SORT */}
              <div className="relative">
                <SlidersHorizontal
                  size={17}
                  className="pointer-events-none absolute left-4 top-1/2 -translate-y-1/2 text-slate-400"
                />

                <select
                  value={sortBy}
                  onChange={(event) =>
                    setSortBy(event.target.value)
                  }
                  className="appearance-none rounded-xl border border-slate-200 bg-white py-3 pl-11 pr-10 text-sm font-medium text-slate-600 outline-none focus:border-blue-400 focus:ring-4 focus:ring-blue-100"
                >
                  <option value="utilization">
                    Utilization
                  </option>

                  <option value="cost">
                    Monthly Cost
                  </option>

                  <option value="savings">
                    Potential Savings
                  </option>

                  <option value="uptime">
                    Uptime
                  </option>
                </select>

                <ChevronDown
                  size={16}
                  className="pointer-events-none absolute right-4 top-1/2 -translate-y-1/2 text-slate-400"
                />
              </div>
            </div>
          </div>

          {/* RESULT COUNT */}
          <div className="mt-6 flex items-center justify-between">
            <p className="text-sm text-slate-500">
              Showing{" "}
              <span className="font-bold text-slate-700">
                {filteredServices.length}
              </span>{" "}
              of{" "}
              <span className="font-bold text-slate-700">
                {services.length}
              </span>{" "}
              services
            </p>

            <span className="hidden items-center gap-2 text-xs font-medium text-slate-400 md:flex">
              <Activity size={15} />
              Monitoring active
            </span>
          </div>
        </section>

        {/* =================================================== */}
        {/* SERVICE GRID */}
        {/* =================================================== */}

        {loadError && (
          <div className="mt-6 rounded-2xl border border-red-200 bg-red-50 p-4 text-sm text-red-700">
            <p className="font-bold">Unable to load infrastructure data</p>
            <p className="mt-1">{loadError}</p>
            <button
              type="button"
              onClick={() => {
                setIsLoading(true);
                fetchInfrastructure();
              }}
              className="mt-3 rounded-lg bg-red-600 px-4 py-2 text-xs font-bold text-white hover:bg-red-700"
            >
              Try Again
            </button>
          </div>
        )}

        {isLoading ? (
          <div className="mt-6 rounded-3xl border border-slate-200 bg-white p-16 text-center text-sm text-slate-500">
            Loading infrastructure data from your applications...
          </div>
        ) : filteredServices.length > 0 ? (
          <div className="mt-6 grid grid-cols-1 gap-6 xl:grid-cols-2">
            {filteredServices.map((service) => {
              const ServiceIcon = service.icon;

              const utilizationStyle =
                getUtilizationStyle(service.utilization);

              return (
                <div
                  key={service.id}
                  className="group overflow-hidden rounded-3xl border border-slate-200 bg-white shadow-sm transition duration-300 hover:-translate-y-1 hover:border-blue-200 hover:shadow-xl"
                >
                  {/* SERVICE HEADER */}
                  <div className="border-b border-slate-100 p-6">
                    <div className="flex items-start justify-between gap-4">
                      <div className="flex items-start gap-4">
                        <div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-gradient-to-br from-blue-600 to-violet-600 text-white shadow-lg shadow-blue-100">
                          <ServiceIcon size={25} />
                        </div>

                        <div>
                          <div className="flex flex-wrap items-center gap-2">
                            <h3 className="text-lg font-bold text-slate-800">
                              {service.name}
                            </h3>

                            <span
                              className={`rounded-full border px-2.5 py-1 text-[10px] font-bold ${getStatusStyle(
                                service.status
                              )}`}
                            >
                              {service.status}
                            </span>
                          </div>

                          <p className="mt-1 text-sm text-slate-500">
                            {service.application}
                          </p>

                          <div className="mt-3 flex flex-wrap gap-2">
                            <span className="rounded-lg bg-slate-100 px-3 py-1.5 text-[10px] font-bold text-slate-600">
                              {service.provider}
                            </span>

                            <span className="rounded-lg bg-blue-50 px-3 py-1.5 text-[10px] font-bold text-blue-600">
                              {service.type}
                            </span>

                            <span className="rounded-lg bg-violet-50 px-3 py-1.5 text-[10px] font-bold text-violet-600">
                              {service.region}
                            </span>
                          </div>
                        </div>
                      </div>

                      <div
                        className={`rounded-xl border px-3 py-2 text-center ${getOptimizationStyle(
                          service.optimization
                        )}`}
                      >
                        <p className="text-[9px] font-bold uppercase tracking-wider">
                          AI
                        </p>

                        <p className="mt-0.5 text-xs font-extrabold">
                          {service.optimization}
                        </p>
                      </div>
                    </div>
                  </div>

                  {/* SERVICE CONTENT */}
                  <div className="p-6">
                    {/* COST */}
                    <div className="flex flex-col gap-4 rounded-2xl bg-slate-50 p-4 sm:flex-row sm:items-center sm:justify-between">
                      <div>
                        <p className="text-xs text-slate-400">
                          Monthly Cost
                        </p>

                        <p className="mt-1 text-2xl font-bold text-slate-800">
                          ${service.monthlyCost.toFixed(2)}
                        </p>
                      </div>

                      <div className="text-left sm:text-right">
                        <p className="text-xs text-slate-400">
                          Potential Savings
                        </p>

                        <p className="mt-1 flex items-center gap-1 text-lg font-bold text-emerald-600">
                          <TrendingDown size={17} />
                          ${service.potentialSavings.toFixed(2)}
                        </p>
                      </div>
                    </div>

                    {/* UTILIZATION */}
                    <div className="mt-6">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <Activity
                            size={16}
                            className={
                              utilizationStyle.text
                            }
                          />

                          <p className="text-xs font-bold text-slate-600">
                            Overall Utilization
                          </p>
                        </div>

                        <span
                          className={`text-sm font-bold ${utilizationStyle.text}`}
                        >
                          {service.utilization === null ? "Not available" : `${service.utilization}%`}
                        </span>
                      </div>

                      <div className="mt-3 h-2.5 overflow-hidden rounded-full bg-slate-100">
                        <div
                          className={`h-full rounded-full transition-all duration-700 ${utilizationStyle.bar}`}
                          style={{
                            width: `${service.utilization ?? 0}%`,
                          }}
                        />
                      </div>
                    </div>

                    {/* RESOURCE GRID */}
                    <div className="mt-6 grid grid-cols-2 gap-3 sm:grid-cols-4">
                      <ResourceMetric
                        icon={Cpu}
                        label="CPU"
                        value={service.cpu}
                      />

                      <ResourceMetric
                        icon={Activity}
                        label="Memory"
                        value={service.memory}
                      />

                      <ResourceMetric
                        icon={HardDrive}
                        label="Storage"
                        value={service.storage}
                      />

                      <ResourceMetric
                        icon={Network}
                        label="Network"
                        value={service.network}
                      />
                    </div>

                    {/* SERVICE DETAILS */}
                    <div className="mt-6 grid grid-cols-2 gap-3">
                      <DetailBox
                        label="Instance"
                        value={
                          service.instance || "Managed Service"
                        }
                      />

                      <DetailBox
                        label="Uptime"
                        value={service.uptime === null || service.uptime === undefined ? "Not available" : `${service.uptime}%`}
                      />
                    </div>

                    {/* AI RECOMMENDATION */}
                    <div className="mt-6 rounded-2xl border border-violet-100 bg-gradient-to-r from-violet-50 to-blue-50 p-4">
                      <div className="flex gap-3">
                        <div className="rounded-xl bg-white p-2.5 text-violet-600 shadow-sm">
                          <Sparkles size={18} />
                        </div>

                        <div>
                          <p className="text-xs font-bold text-slate-700">
                            AI Recommendation
                          </p>

                          <p className="mt-1 text-xs leading-relaxed text-slate-500">
                            {service.recommendation}
                          </p>
                        </div>
                      </div>
                    </div>

                    {/* FOOTER */}
                    <div className="mt-6 flex items-center justify-between border-t border-slate-100 pt-5">
                      <div className="flex items-center gap-2">
                        <span className="h-2 w-2 rounded-full bg-emerald-500" />

                        <span className="text-xs font-medium text-slate-400">
                          Data available
                        </span>
                      </div>

                      <button
                        type="button"
                        onClick={() =>
                          handleViewResource(service)
                        }
                        className="flex items-center gap-2 rounded-xl bg-slate-900 px-4 py-2.5 text-xs font-bold text-white transition hover:bg-blue-600"
                      >
                        View Resource
                        <ArrowUpRight size={15} />
                      </button>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        ) : (
          <div className="mt-6 flex flex-col items-center justify-center rounded-3xl border border-dashed border-slate-300 bg-white p-16 text-center">
            <div className="flex h-16 w-16 items-center justify-center rounded-2xl bg-slate-100 text-slate-400">
              <Search size={28} />
            </div>

            <h3 className="mt-5 text-lg font-bold text-slate-700">
              No Cloud Services Found
            </h3>

            <p className="mt-2 max-w-sm text-sm text-slate-500">
              {services.length === 0
                ? "Add an application first to see its recorded infrastructure details here."
                : "Try changing your search or filter settings."}
            </p>

            <button
              type="button"
              onClick={() => {
                setSearchQuery("");
                setProviderFilter("All");
                setStatusFilter("All");
                setSortBy("utilization");
              }}
              className="mt-6 rounded-xl bg-slate-900 px-5 py-3 text-sm font-bold text-white transition hover:bg-blue-600"
            >
              Reset Filters
            </button>
          </div>
        )}

        {/* =================================================== */}
        {/* AI INFRASTRUCTURE SUMMARY */}
        {/* =================================================== */}

        <section className="mt-10 overflow-hidden rounded-3xl border border-blue-100 bg-gradient-to-r from-blue-50 via-white to-violet-50 p-8">
          <div className="flex flex-col gap-7 lg:flex-row lg:items-center lg:justify-between">
            <div className="flex items-start gap-5">
              <div className="rounded-2xl bg-gradient-to-br from-blue-600 to-violet-600 p-4 text-white shadow-lg">
                <BrainIcon />
              </div>

              <div>
                <p className="text-xs font-bold uppercase tracking-[0.2em] text-violet-600">
                  CloudMorph Infrastructure Intelligence
                </p>

                <h2 className="mt-3 text-2xl font-bold text-slate-800">
                  Infrastructure optimization overview
                </h2>

                <p className="mt-3 max-w-2xl text-sm leading-relaxed text-slate-500">
                  CloudMorph summarizes recorded application utilization, saved cost estimates, and optimization decisions. Live infrastructure telemetry and automated monitoring require a connected cloud-provider integration.
                </p>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-3">
              <SummaryMetric
                label="Services"
                value={services.length}
              />

              <SummaryMetric
                label="Savings"
                value={`$${totalPotentialSavings.toFixed(2)}`}
              />

              <SummaryMetric
                label="Healthy"
                value={`${healthyServices}/${services.length}`}
              />

              <SummaryMetric
                label="Utilization"
                value={averageUtilization === null ? "—" : `${averageUtilization}%`}
              />
            </div>
          </div>
        </section>

        <div className="h-6" />
      </main>

      {/* ===================================================== */}
      {/* RESOURCE DETAILS MODAL */}
      {/* ===================================================== */}

      {selectedResource && (
        <ResourceDetailsModal
          resource={selectedResource}
          onClose={handleCloseResource}
          getStatusStyle={getStatusStyle}
          getOptimizationStyle={getOptimizationStyle}
        />
      )}
    </div>
  );
}

/* ========================================================= */
/* RESOURCE DETAILS MODAL */
/* ========================================================= */

function ResourceDetailsModal({
  resource,
  onClose,
  getStatusStyle,
  getOptimizationStyle,
}) {
  const ResourceIcon = resource.icon;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/60 p-4 backdrop-blur-sm"
      onClick={onClose}
    >
      <div
        className="max-h-[90vh] w-full max-w-3xl overflow-y-auto rounded-3xl border border-slate-200 bg-white shadow-2xl"
        onClick={(event) => event.stopPropagation()}
      >
        {/* MODAL HEADER */}
        <div className="sticky top-0 z-10 border-b border-slate-100 bg-white/95 p-6 backdrop-blur">
          <div className="flex items-start justify-between gap-4">
            <div className="flex items-start gap-4">
              <div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-gradient-to-br from-blue-600 to-violet-600 text-white shadow-lg">
                <ResourceIcon size={25} />
              </div>

              <div>
                <div className="flex flex-wrap items-center gap-2">
                  <h2 className="text-xl font-bold text-slate-800">
                    {resource.name}
                  </h2>

                  <span
                    className={`rounded-full border px-2.5 py-1 text-[10px] font-bold ${getStatusStyle(
                      resource.status
                    )}`}
                  >
                    {resource.status}
                  </span>
                </div>

                <p className="mt-1 text-sm text-slate-500">
                  {resource.application}
                </p>

                <div className="mt-3 flex flex-wrap gap-2">
                  <span className="rounded-lg bg-slate-100 px-3 py-1.5 text-[10px] font-bold text-slate-600">
                    {resource.provider}
                  </span>

                  <span className="rounded-lg bg-blue-50 px-3 py-1.5 text-[10px] font-bold text-blue-600">
                    {resource.type}
                  </span>

                  <span className="rounded-lg bg-violet-50 px-3 py-1.5 text-[10px] font-bold text-violet-600">
                    {resource.region}
                  </span>
                </div>
              </div>
            </div>

            <button
              type="button"
              onClick={onClose}
              className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-slate-100 text-slate-500 transition hover:bg-slate-200 hover:text-slate-800"
              aria-label="Close resource details"
            >
              <X size={19} />
            </button>
          </div>
        </div>

        {/* MODAL CONTENT */}
        <div className="space-y-6 p-6">
          {/* RESOURCE SUMMARY */}
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
            <ResourceSummaryCard
              label="Monthly Cost"
              value={`$${resource.monthlyCost.toFixed(2)}`}
              icon={DollarSign}
              iconStyle="bg-blue-50 text-blue-600"
            />

            <ResourceSummaryCard
              label="Potential Savings"
              value={`$${resource.potentialSavings.toFixed(2)}`}
              icon={TrendingDown}
              iconStyle="bg-emerald-50 text-emerald-600"
            />

            <ResourceSummaryCard
              label="Uptime"
              value={resource.uptime === null || resource.uptime === undefined ? "Not available" : `${resource.uptime}%`}
              icon={Activity}
              iconStyle="bg-violet-50 text-violet-600"
            />
          </div>

          {/* AI DECISION */}
          <div className="rounded-2xl border border-violet-100 bg-gradient-to-r from-violet-50 to-blue-50 p-5">
            <div className="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
              <div className="flex items-start gap-3">
                <div className="rounded-xl bg-white p-3 text-violet-600 shadow-sm">
                  <BrainCircuit size={20} />
                </div>

                <div>
                  <p className="text-xs font-bold uppercase tracking-wider text-violet-600">
                    CloudMorph AI Decision
                  </p>

                  <h3 className="mt-1 text-lg font-bold text-slate-800">
                    {resource.optimization}
                  </h3>

                  <p className="mt-1 text-sm leading-relaxed text-slate-500">
                    Current infrastructure recommendation based on
                    utilization, cost, and service configuration.
                  </p>
                </div>
              </div>

              <span
                className={`self-start rounded-xl border px-4 py-2 text-xs font-extrabold ${getOptimizationStyle(
                  resource.optimization
                )}`}
              >
                {resource.optimization}
              </span>
            </div>

            <div className="mt-4 rounded-xl bg-white/80 p-4">
              <p className="text-xs font-bold text-slate-700">
                Recommendation
              </p>

              <p className="mt-1 text-sm leading-relaxed text-slate-500">
                {resource.recommendation}
              </p>
            </div>
          </div>

          {/* RESOURCE CONFIGURATION */}
          <section>
            <div className="flex items-center gap-2">
              <Server
                size={20}
                className="text-blue-600"
              />

              <h3 className="text-lg font-bold text-slate-800">
                Resource Configuration
              </h3>
            </div>

            <div className="mt-4 grid grid-cols-1 gap-3 sm:grid-cols-2">
              <ModalDetail
                label="Service"
                value={resource.name}
              />

              <ModalDetail
                label="Application"
                value={resource.application}
              />

              <ModalDetail
                label="Provider"
                value={resource.provider}
              />

              <ModalDetail
                label="Region"
                value={resource.region}
              />

              <ModalDetail
                label="Service Type"
                value={resource.type}
              />

              <ModalDetail
                label="Instance / Configuration"
                value={
                  resource.instance || "Managed Service"
                }
              />
            </div>
          </section>

          {/* RESOURCE METRICS */}
          <section>
            <div className="flex items-center gap-2">
              <BarChart3
                size={20}
                className="text-violet-600"
              />

              <h3 className="text-lg font-bold text-slate-800">
                Resource Metrics
              </h3>
            </div>

            <div className="mt-4 grid grid-cols-1 gap-4 sm:grid-cols-2">
              <ModalMetric
                icon={Activity}
                label="Overall Utilization"
                value={resource.utilization}
              />

              <ModalMetric
                icon={Cpu}
                label="CPU"
                value={resource.cpu}
              />

              <ModalMetric
                icon={Activity}
                label="Memory"
                value={resource.memory}
              />

              <ModalMetric
                icon={HardDrive}
                label="Storage"
                value={resource.storage}
              />

              <ModalMetric
                icon={Network}
                label="Network"
                value={resource.network}
              />

              <ModalMetric
                icon={ShieldCheck}
                label="Uptime"
                value={resource.uptime}
                suffix="%"
              />
            </div>
          </section>

          {/* MONITORING STATUS */}
          <div className="rounded-2xl border border-emerald-100 bg-emerald-50 p-5">
            <div className="flex items-start gap-3">
              <div className="rounded-xl bg-white p-3 text-emerald-600 shadow-sm">
                <Check size={20} />
              </div>

              <div>
                <p className="text-sm font-bold text-emerald-700">
                  Recorded Resource Information
                </p>

                <p className="mt-1 text-xs leading-relaxed text-emerald-600">
                  This view uses application data and saved estimates returned by the backend. Live CPU, memory, storage, network, and uptime telemetry appears only when a connected provider supplies it.
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* MODAL FOOTER */}
        <div className="border-t border-slate-100 bg-slate-50 p-5">
          <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
            <p className="text-xs text-slate-400">
              Resource details reflect application records and any metrics supplied by the backend; unavailable telemetry is shown as not available.
            </p>

            <button
              type="button"
              onClick={onClose}
              className="rounded-xl bg-slate-900 px-5 py-2.5 text-xs font-bold text-white transition hover:bg-blue-600"
            >
              Close
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

/* ========================================================= */
/* RESOURCE SUMMARY CARD */
/* ========================================================= */

function ResourceSummaryCard({
  label,
  value,
  icon: Icon,
  iconStyle,
}) {
  return (
    <div className="rounded-2xl border border-slate-100 bg-slate-50 p-4">
      <div
        className={`inline-flex rounded-xl p-3 ${iconStyle}`}
      >
        <Icon size={18} />
      </div>

      <p className="mt-4 text-xs font-medium text-slate-400">
        {label}
      </p>

      <p className="mt-1 text-xl font-bold text-slate-800">
        {value}
      </p>
    </div>
  );
}

/* ========================================================= */
/* MODAL DETAIL */
/* ========================================================= */

function ModalDetail({
  label,
  value,
}) {
  return (
    <div className="rounded-xl border border-slate-100 bg-slate-50 p-4">
      <p className="text-[10px] font-medium uppercase tracking-wide text-slate-400">
        {label}
      </p>

      <p className="mt-1 truncate text-sm font-bold text-slate-700">
        {value}
      </p>
    </div>
  );
}

/* ========================================================= */
/* MODAL METRIC */
/* ========================================================= */

function ModalMetric({
  icon: Icon,
  label,
  value,
  suffix = "%",
}) {
  const numericValue =
    value === null || value === undefined || value === ""
      ? null
      : Number(value);
  const hasValidValue =
    Number.isFinite(numericValue) && numericValue >= 0 && numericValue <= 100;

  const getColor = () => {
    if (!hasValidValue) {
      return {
        text: "text-slate-300",
        bar: "bg-slate-200",
      };
    }

    if (numericValue >= 85) {
      return {
        text: "text-red-500",
        bar: "bg-red-500",
      };
    }

    if (numericValue >= 70) {
      return {
        text: "text-amber-500",
        bar: "bg-amber-500",
      };
    }

    if (numericValue === 0) {
      return {
        text: "text-slate-300",
        bar: "bg-slate-200",
      };
    }

    return {
      text: "text-emerald-500",
      bar: "bg-emerald-500",
    };
  };

  const color = getColor();

  return (
    <div className="rounded-2xl border border-slate-100 bg-white p-4 shadow-sm">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Icon
            size={16}
            className={color.text}
          />

          <span className="text-xs font-bold text-slate-600">
            {label}
          </span>
        </div>

        <span
          className={`text-sm font-bold ${color.text}`}
        >
          {!hasValidValue ? "Not available" : `${numericValue}${suffix}`}
        </span>
      </div>

      <div className="mt-3 h-2 overflow-hidden rounded-full bg-slate-100">
        <div
          className={`h-full rounded-full transition-all duration-500 ${color.bar}`}
          style={{
            width: `${hasValidValue ? numericValue : 0}%`,
          }}
        />
      </div>
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
/* OVERVIEW CARD */
/* ========================================================= */

function OverviewCard({
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
/* UTILIZATION CARD */
/* ========================================================= */

function UtilizationCard({
  icon: Icon,
  label,
  value,
  description,
}) {
  const getColor = () => {
    if (value >= 85) {
      return "bg-red-500";
    }

    if (value >= 70) {
      return "bg-amber-500";
    }

    return "bg-emerald-500";
  };

  return (
    <div className="rounded-2xl border border-slate-100 bg-slate-50 p-5">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="rounded-xl bg-white p-3 text-slate-500 shadow-sm">
            <Icon size={19} />
          </div>

          <div>
            <p className="text-sm font-bold text-slate-700">
              {label}
            </p>

            <p className="text-[10px] text-slate-400">
              {description}
            </p>
          </div>
        </div>

        <span className="text-lg font-bold text-slate-700">
          {value === null || value === undefined ? "—" : `${value}%`}
        </span>
      </div>

      <div className="mt-5 h-2 overflow-hidden rounded-full bg-white">
        <div
          className={`h-full rounded-full ${getColor()}`}
          style={{
            width: `${value ?? 0}%`,
          }}
        />
      </div>
    </div>
  );
}

/* ========================================================= */
/* RESOURCE METRIC */
/* ========================================================= */

function ResourceMetric({
  icon: Icon,
  label,
  value,
}) {
  const getColor = () => {
    if (value === null || value === undefined) {
      return "text-slate-300";
    }

    if (value >= 85) {
      return "text-red-500";
    }

    if (value >= 70) {
      return "text-amber-500";
    }

    if (value === 0) {
      return "text-slate-300";
    }

    return "text-emerald-500";
  };

  return (
    <div className="rounded-xl border border-slate-100 bg-slate-50 p-3">
      <div className="flex items-center justify-between">
        <Icon
          size={14}
          className={getColor()}
        />

        <span className="text-xs font-bold text-slate-600">
          {value === null || value === undefined ? "—" : `${value}%`}
        </span>
      </div>

      <p className="mt-2 text-[10px] font-medium text-slate-400">
        {label}
      </p>

      <div className="mt-2 h-1.5 overflow-hidden rounded-full bg-white">
        <div
          className={`h-full rounded-full ${
            value >= 85
              ? "bg-red-500"
              : value >= 70
              ? "bg-amber-500"
              : value === 0
              ? "bg-slate-200"
              : "bg-emerald-500"
          }`}
          style={{
            width: `${value ?? 0}%`,
          }}
        />
      </div>
    </div>
  );
}

/* ========================================================= */
/* DETAIL BOX */
/* ========================================================= */

function DetailBox({
  label,
  value,
}) {
  return (
    <div className="rounded-xl bg-slate-50 p-4">
      <p className="text-[10px] font-medium text-slate-400">
        {label}
      </p>

      <p className="mt-1 truncate text-xs font-bold text-slate-700">
        {value}
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
/* BRAIN ICON */
/* ========================================================= */

function BrainIcon() {
  return <BrainCircuit size={28} />;
}

export default Infrastructure;