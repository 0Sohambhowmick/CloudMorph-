import { useCallback, useEffect, useMemo, useState } from "react";
import { Link, useParams } from "react-router-dom";
import {
  ArrowLeft,
  Cloud,
  DollarSign,
  TrendingDown,
  Gauge,
  BrainCircuit,
  Server,
  Database,
  Cpu,
  ShieldCheck,
  Sparkles,
  Globe,
  Activity,
  Clock3,
  Zap,
  ArrowUpRight,
  RefreshCw,
  AlertTriangle,
} from "lucide-react";
import API from "../services/api";

const toFiniteNumber = (value, fallback = null) => {
  if (value === null || value === undefined || value === "") return fallback;
  const number = Number(value);
  return Number.isFinite(number) ? number : fallback;
};

const getApplicationFromResponse = (data) =>
  data?.application ?? data?.data?.application ?? data?.data ?? data;

const normalizeApplication = (data) => {
  if (!data || typeof data !== "object" || Array.isArray(data)) return null;
  const id = data.id ?? data._id;
  if (id === null || id === undefined || String(id).trim() === "") return null;

  const cost = Math.max(
    0,
    toFiniteNumber(data.monthlyCost ?? data.monthly_cost ?? data.cost, 0)
  );
  const optimizedCostValue = toFiniteNumber(
    data.optimized_cost ?? data.optimizedCost ?? data.optimized_cost_monthly,
    null
  );
  const utilizationValue = toFiniteNumber(
    data.utilization ?? data.usagePercent ?? data.utilization_percent ?? data.utilizationPercent,
    null
  );
  const normalizedUtilization =
    utilizationValue === null
      ? null
      : Math.round(
          Math.min(
            100,
            Math.max(0, utilizationValue > 0 && utilizationValue <= 1 ? utilizationValue * 100 : utilizationValue)
          )
        );

  const decision = String(data.decision ?? "ANALYZE").toUpperCase();
  const status = data.status || (decision === "SWITCH" || decision === "RESIZE" ? "Warning" : "Active");

  return {
    ...data,
    id: String(id),
    name: data.name || "Unnamed Application",
    provider: data.provider || "AWS",
    cost,
    optimizedCost:
      optimizedCostValue === null
        ? null
        : Math.max(0, optimizedCostValue),
    utilization: normalizedUtilization,
    cpuUtilization: toFiniteNumber(data.cpu_utilization ?? data.cpuUtilization ?? data.cpu_usage, null),
    memoryUtilization: toFiniteNumber(data.memory_utilization ?? data.memoryUtilization ?? data.memory_usage, null),
    networkUtilization: toFiniteNumber(data.network_utilization ?? data.networkUtilization ?? data.network_usage, null),
    healthScore: toFiniteNumber(data.health_score ?? data.healthScore, null),
    decision,
    status,
    region: data.region || "Not configured",
    description: data.description || "No description provided.",
    category: data.category || data.service || "Cloud Application",
    services: Math.max(0, toFiniteNumber(data.services ?? data.service_count, 1)),
    recommendation: data.recommendation || data.reasons || null,
    updatedAt: data.updated_at ?? data.updatedAt ?? null,
    createdAt: data.created_at ?? data.createdAt ?? null,
  };
};

const formatCurrency = (value) =>
  value === null || value === undefined || !Number.isFinite(Number(value))
    ? "Not available"
    : `$${Number(value).toFixed(2)}`;

const formatPercent = (value) =>
  value === null || value === undefined || !Number.isFinite(Number(value))
    ? "Not available"
    : `${Math.round(Math.min(100, Math.max(0, Number(value))))}%`;

const formatDate = (value) => {
  if (!value) return "Not available";
  const date = new Date(value);
  return Number.isNaN(date.getTime())
    ? "Not available"
    : date.toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
};

function ApplicationDetails() {
  const { id } = useParams();
  const [application, setApplication] = useState(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState("");

  const fetchApplication = useCallback(async ({ showLoader = true } = {}) => {
    if (!id || !String(id).trim()) {
      setApplication(null);
      setError("No application ID was provided.");
      setLoading(false);
      return;
    }

    if (showLoader) setLoading(true);
    setError("");

    try {
      const response = await API.get(`/applications/${encodeURIComponent(id)}`);
      const data = getApplicationFromResponse(response.data);
      const normalized = normalizeApplication(data);
      if (!normalized) {
        throw new Error("The server returned an invalid application record.");
      }
      setApplication(normalized);
    } catch (err) {
      setApplication(null);
      if (err.response?.status === 401) {
        setError("Your session has expired. Please log in again.");
      } else if (err.response?.status === 404) {
        setError("The requested application doesn't exist or has been removed.");
      } else if (err.response?.status === 400) {
        setError("The application ID is invalid.");
      } else if (err.response?.data?.detail) {
        setError(String(err.response.data.detail));
      } else if (err.response) {
        setError("Unable to load application details. Please try again.");
      } else if (err.message === "The server returned an invalid application record.") {
        setError(err.message);
      } else {
        setError("Unable to connect to CloudMorph API. Check that the backend is running.");
      }
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, [id]);

  useEffect(() => {
    let active = true;
    const run = async () => {
      // Prevent stale responses from replacing data after the route changes.
      if (!active) return;
      await fetchApplication({ showLoader: true });
    };
    run();
    return () => {
      active = false;
    };
  }, [fetchApplication]);

  const handleRefresh = async () => {
    if (refreshing) return;
    setRefreshing(true);
    await fetchApplication({ showLoader: false });
  };

  const metrics = useMemo(() => {
    const currentCost = Math.max(0, toFiniteNumber(application?.cost, 0));
    const optimizedCost = application?.optimizedCost;
    const hasOptimizedCost = optimizedCost !== null && optimizedCost !== undefined;
    const savings = hasOptimizedCost ? Math.max(0, currentCost - optimizedCost) : null;
    const utilization = application?.utilization ?? null;
    const healthScore = application?.healthScore ?? null;
    return { currentCost, optimizedCost, hasOptimizedCost, savings, utilization, healthScore };
  }, [application]);

  if (loading) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-slate-100 p-8">
        <div className="rounded-3xl border border-slate-200 bg-white p-10 text-center shadow-xl">
          <div className="mx-auto h-12 w-12 animate-spin rounded-full border-4 border-blue-100 border-t-blue-600" />
          <h2 className="mt-5 text-2xl font-bold text-slate-800">Loading Application</h2>
          <p className="mt-3 text-slate-500">Fetching application details from CloudMorph.</p>
        </div>
      </div>
    );
  }

  if (!application) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-slate-100 p-8">
        <div className="max-w-xl rounded-3xl border border-slate-200 bg-white p-10 text-center shadow-xl">
          <Cloud size={52} className="mx-auto text-blue-600" />
          <h2 className="mt-5 text-3xl font-bold text-slate-800">Unable to Load Application</h2>
          <p role="alert" className="mt-3 text-slate-500">
            {error || "The requested application doesn't exist or has been removed."}
          </p>
          <div className="mt-7 flex flex-wrap justify-center gap-3">
            <button
              type="button"
              onClick={() => fetchApplication({ showLoader: true })}
              className="inline-flex items-center justify-center gap-2 rounded-xl border border-slate-300 px-6 py-3 font-semibold text-slate-700 transition hover:bg-slate-100"
            >
              <RefreshCw size={17} /> Try Again
            </button>
            <Link to="/app/applications" className="inline-flex items-center gap-2 rounded-xl bg-blue-600 px-6 py-3 font-semibold text-white transition hover:bg-blue-700">
              <ArrowLeft size={18} /> Back to Applications
            </Link>
          </div>
        </div>
      </div>
    );
  }

  const { currentCost, optimizedCost, hasOptimizedCost, savings, utilization, healthScore } = metrics;
  const utilizationWidth = utilization ?? 0;
  const statusIsWarning = String(application.status).toLowerCase().includes("warning") || String(application.status).toLowerCase().includes("error");
  const resources = [
    { name: "Compute Instances", icon: Cpu, value: application.compute_instances ?? application.computeInstances ?? null, status: "Resource data" },
    { name: "Object Storage", icon: Database, value: application.storage_resources ?? application.storageResources ?? null, status: "Resource data" },
    { name: "Serverless", icon: Sparkles, value: application.serverless_functions ?? application.serverlessFunctions ?? null, status: "Resource data" },
    { name: "Cloud Services", icon: ShieldCheck, value: application.services, status: "Configured" },
  ];

  const timeline = [
    { time: formatDate(application.updatedAt), title: "Application record last updated", color: "bg-blue-500" },
    { time: formatDate(application.createdAt), title: "Application record created", color: "bg-violet-500" },
  ];

  return (
    <div className="min-h-screen bg-slate-100">
      <section className="relative overflow-hidden bg-gradient-to-br from-slate-950 via-blue-950 to-violet-950 px-8 py-10 text-white">
        <div className="absolute -left-24 top-0 h-72 w-72 rounded-full bg-blue-500/20 blur-3xl" />
        <div className="absolute -right-24 bottom-0 h-72 w-72 rounded-full bg-violet-500/20 blur-3xl" />
        <div className="relative">
          <Link to="/app/applications" className="mb-8 inline-flex items-center gap-2 rounded-xl border border-white/10 bg-white/10 px-4 py-2 text-sm backdrop-blur transition hover:bg-white/20">
            <ArrowLeft size={18} /> Back
          </Link>
          <div className="flex flex-col gap-8 lg:flex-row lg:items-center lg:justify-between">
            <div className="min-w-0">
              <div className="flex items-center gap-4">
                <div className="rounded-2xl bg-white/10 p-4 backdrop-blur"><Cloud size={34} className="text-blue-300" /></div>
                <div className="min-w-0">
                  <h1 className="break-words text-3xl font-bold sm:text-4xl">{application.name}</h1>
                  <p className="mt-2 text-slate-300">{application.provider} • {application.category}</p>
                </div>
              </div>
              <p className="mt-4 max-w-2xl text-sm leading-relaxed text-slate-300">{application.description}</p>
              <div className="mt-6 flex flex-wrap gap-3">
                <span className="rounded-full bg-blue-500/20 px-4 py-2 text-sm font-semibold text-blue-200">{application.provider}</span>
                <span className="rounded-full bg-violet-500/20 px-4 py-2 text-sm font-semibold text-violet-200">{application.decision}</span>
                <span className={`rounded-full px-4 py-2 text-sm font-semibold ${statusIsWarning ? "bg-amber-500/20 text-amber-200" : "bg-white/10 text-slate-200"}`}>{application.status}</span>
              </div>
            </div>
            <div className="rounded-3xl border border-white/10 bg-white/10 p-6 backdrop-blur">
              <div className="flex flex-col items-center">
                <div className="relative flex h-28 w-28 items-center justify-center rounded-full border-[8px] border-white/15">
                  {healthScore !== null && <div className="absolute inset-0 rounded-full border-[8px] border-transparent border-t-emerald-400" style={{ transform: `rotate(${Math.min(100, Math.max(0, healthScore)) * 3.6}deg)` }} />}
                  <span className="text-3xl font-bold">{healthScore === null ? "—" : Math.round(healthScore)}</span>
                </div>
                <p className="mt-4 text-sm text-slate-300">Health Score</p>
                <p className="mt-1 text-xs text-slate-400">{healthScore === null ? "No health score provided" : "Based on saved application data"}</p>
              </div>
            </div>
          </div>
          <div className="mt-6 flex justify-end">
            <button type="button" onClick={handleRefresh} disabled={refreshing} className="inline-flex items-center gap-2 rounded-xl border border-white/15 bg-white/10 px-4 py-2.5 text-sm font-semibold text-white transition hover:bg-white/20 disabled:cursor-not-allowed disabled:opacity-60">
              <RefreshCw size={16} className={refreshing ? "animate-spin" : ""} /> {refreshing ? "Refreshing..." : "Refresh Details"}
            </button>
          </div>
        </div>
      </section>

      <main className="space-y-8 p-8">
        {error && <div role="alert" className="rounded-xl border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-800"><AlertTriangle size={16} className="mr-2 inline" />{error}</div>}
        <section className="grid grid-cols-1 gap-5 md:grid-cols-2 xl:grid-cols-4">
          <StatCard icon={DollarSign} title="Current Cost" value={formatCurrency(currentCost)} color="text-blue-600" />
          <StatCard icon={TrendingDown} title="Optimized Cost" value={hasOptimizedCost ? formatCurrency(optimizedCost) : "Not analyzed"} color="text-emerald-600" />
          <StatCard icon={Sparkles} title="Estimated Monthly Savings" value={savings === null ? "Not available" : formatCurrency(savings)} color="text-violet-600" />
          <StatCard icon={Globe} title="Region" value={application.region} color="text-orange-600" />
        </section>

        <section className="grid gap-8 xl:grid-cols-[1.6fr_1fr]">
          <div className="space-y-8">
            <div className="rounded-3xl border border-slate-200 bg-white p-7 shadow-sm">
              <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
                <div className="flex items-center gap-3"><Gauge className="text-blue-600" size={24} /><div><h2 className="text-xl font-bold text-slate-800">Resource Utilization</h2><p className="text-sm text-slate-500">Saved utilization data</p></div></div>
                <span className="rounded-full bg-blue-50 px-3 py-1 text-sm font-bold text-blue-700">{formatPercent(utilization)}</span>
              </div>
              <div className="mt-8">
                <div className="mb-3 flex justify-between text-sm"><span className="text-slate-500">Overall usage</span><span className="font-semibold">{formatPercent(utilization)}</span></div>
                <div className="h-4 overflow-hidden rounded-full bg-slate-100"><div className="h-full rounded-full bg-gradient-to-r from-blue-500 via-blue-600 to-violet-600 transition-all duration-700" style={{ width: `${utilizationWidth}%` }} /></div>
                {utilization === null && <p className="mt-2 text-xs text-slate-400">Utilization telemetry has not been provided for this application.</p>}
                <div className="mt-6 grid grid-cols-1 gap-4 sm:grid-cols-3">
                  <MiniStat label="CPU" value={formatPercent(application.cpuUtilization)} />
                  <MiniStat label="Memory" value={formatPercent(application.memoryUtilization)} />
                  <MiniStat label="Network" value={formatPercent(application.networkUtilization)} />
                </div>
              </div>
            </div>

            <div className="rounded-3xl border border-slate-200 bg-white p-7 shadow-sm">
              <div className="flex items-center gap-3"><Server className="text-blue-600" size={24} /><div><h2 className="text-xl font-bold text-slate-800">Infrastructure Inventory</h2><p className="text-sm text-slate-500">Resource details available in the application record</p></div></div>
              <div className="mt-7 grid grid-cols-1 gap-4 md:grid-cols-2">
                {resources.map((resource) => {
                  const Icon = resource.icon;
                  const hasValue = resource.value !== null && resource.value !== undefined && resource.value !== "";
                  return <div key={resource.name} className="group rounded-2xl border border-slate-200 bg-slate-50 p-5 transition hover:-translate-y-1 hover:border-blue-200 hover:bg-white hover:shadow-lg">
                    <div className="flex items-start justify-between"><div className="flex gap-4"><div className="rounded-xl bg-blue-50 p-3 text-blue-600 transition group-hover:bg-blue-600 group-hover:text-white"><Icon size={22} /></div><div><h3 className="font-bold text-slate-800">{resource.name}</h3><p className="mt-1 text-sm text-slate-500">{hasValue ? String(resource.value) : "Not provided"}</p></div></div><ArrowUpRight size={18} className="text-slate-400" /></div>
                    <div className="mt-5 inline-flex rounded-full bg-slate-200 px-3 py-1 text-xs font-semibold text-slate-600">{hasValue ? resource.status : "No telemetry"}</div>
                  </div>;
                })}
              </div>
              <p className="mt-4 text-xs leading-relaxed text-slate-400">Resource inventory and live status require a connected cloud provider or resource telemetry integration.</p>
            </div>
          </div>

          <div className="space-y-8">
            <div className="overflow-hidden rounded-3xl bg-gradient-to-br from-blue-600 via-blue-700 to-violet-700 text-white shadow-xl">
              <div className="relative p-7"><div className="absolute -right-10 -top-10 h-36 w-36 rounded-full bg-white/10 blur-2xl" />
                <div className="relative flex items-center gap-3"><BrainCircuit size={28} /><div><h2 className="text-xl font-bold">AI Optimization</h2><p className="text-sm text-blue-100">Saved recommendation</p></div></div>
                <div className="relative mt-7 rounded-2xl bg-white/10 p-5 backdrop-blur"><p className="text-sm text-blue-100">Recommended Action</p><h3 className="mt-2 break-words text-3xl font-bold">{application.decision}</h3>
                  <p className="mt-4 text-blue-100">{application.recommendation ? String(application.recommendation) : hasOptimizedCost ? <>Estimated monthly savings: <span className="font-bold text-white">{formatCurrency(savings)}</span>, based on the saved optimized-cost estimate.</> : "Run an optimization analysis to generate a recommendation and cost estimate."}</p>
                  <Link to="/app/optimizer" className="mt-6 flex w-full items-center justify-center gap-2 rounded-xl bg-white py-3 font-semibold text-blue-700 transition hover:bg-blue-50"><Zap size={18} /> Open AI Analysis</Link>
                </div>
              </div>
            </div>

            <div className="rounded-3xl border border-slate-200 bg-white p-7 shadow-sm">
              <div className="flex items-center gap-3"><Activity className="text-blue-600" size={24} /><div><h2 className="text-xl font-bold text-slate-800">Application Activity</h2><p className="text-sm text-slate-500">Timestamps stored with this record</p></div></div>
              <div className="mt-7 space-y-6">
                {timeline.map((item) => <div key={item.title} className="flex gap-4"><div className="flex flex-col items-center"><div className={`h-3 w-3 rounded-full ${item.color}`} /></div><div className="pb-2"><p className="font-semibold text-slate-800">{item.title}</p><p className="mt-1 flex items-center gap-1 text-sm text-slate-500"><Clock3 size={14} />{item.time}</p></div></div>)}
              </div>
              <p className="mt-3 text-xs leading-relaxed text-slate-400">This timeline uses application record timestamps; it does not represent live monitoring events.</p>
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}

function StatCard({ icon: Icon, title, value, color }) {
  return <div className="group rounded-3xl border border-slate-200 bg-white p-6 shadow-sm transition hover:-translate-y-1 hover:border-blue-200 hover:shadow-xl"><div className="flex items-center justify-between"><div className="rounded-2xl bg-slate-100 p-3 transition group-hover:bg-blue-50"><Icon size={24} className={color} /></div><ArrowUpRight size={18} className="text-slate-300" /></div><p className="mt-5 text-sm text-slate-500">{title}</p><h3 className="mt-2 break-words text-2xl font-bold text-slate-800 sm:text-3xl">{value}</h3></div>;
}

function MiniStat({ label, value }) {
  return <div className="rounded-xl bg-slate-50 p-4 text-center"><p className="text-xs text-slate-500">{label}</p><p className="mt-1 break-words text-lg font-bold text-slate-800">{value}</p></div>;
}

export default ApplicationDetails;
