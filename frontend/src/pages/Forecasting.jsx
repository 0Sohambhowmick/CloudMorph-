import { useEffect, useState } from "react";
import {
  Activity,
  Cpu,
  MemoryStick,
  RefreshCw,
  Clock,
  TrendingUp,
  AlertCircle,
  CheckCircle2,
  Server,
  BrainCircuit,
  Gauge,
  Layers,
  Database,
  Search,
  Zap,
} from "lucide-react";

import {
  checkForecastHealth,
  getHistoricalData,
  generateForecast,
} from "../services/forecasting";

// --------------------------------------------------
// FORECAST VALIDATION
// --------------------------------------------------

const WINDOW_SIZE = 12;
const FEATURE_COUNT = 6;
const FORECAST_HORIZONS = ["15min", "30min"];

const isFiniteNumber = (value) => {
  if (typeof value === "boolean" || value === null || value === "") {
    return false;
  }

  return Number.isFinite(Number(value));
};

const validateHistoricalData = (history) => {
  if (!Array.isArray(history) || history.length !== WINDOW_SIZE) {
    throw new Error(
      "Historical data must contain exactly 12 observations."
    );
  }

  history.forEach((record, index) => {
    if (!Array.isArray(record) || record.length !== FEATURE_COUNT) {
      throw new Error(
        `Historical observation ${index + 1} must contain exactly six features.`
      );
    }

    record.forEach((value, featureIndex) => {
      if (!isFiniteNumber(value)) {
        throw new Error(
          `Historical observation ${index + 1} contains a missing or non-numeric feature.`
        );
      }

      const number = Number(value);

      // Feature order:
      // 0: CPU, 1: Memory, 2: Disk,
      // 3: Load1, 4: Load5, 5: Load15

      if (featureIndex <= 2 && (number < 0 || number > 100)) {
        throw new Error(
          `CPU, memory and disk utilization must be between 0 and 100. Invalid value in observation ${index + 1}.`
        );
      }

      if (featureIndex >= 3 && number < 0) {
        throw new Error(
          `Load values cannot be negative. Invalid value in observation ${index + 1}.`
        );
      }
    });
  });

  return true;
};

const validateForecastResponse = (result) => {
  if (
    !result ||
    result.status !== "success" ||
    !result.forecasts ||
    typeof result.forecasts !== "object"
  ) {
    throw new Error(
      "The forecasting API returned an invalid response."
    );
  }

  FORECAST_HORIZONS.forEach((horizon) => {
    const prediction = result.forecasts[horizon];

    if (!prediction || typeof prediction !== "object") {
      throw new Error(
        `The forecasting API did not return a valid ${horizon} prediction.`
      );
    }

    const fields = [
      {
        label: "CPU forecast",
        value: prediction.cpu_utilization,
      },
      {
        label: "Memory forecast",
        value: prediction.memory_utilization,
      },
    ];

    fields.forEach(({ label, value }) => {
      if (!isFiniteNumber(value)) {
        throw new Error(
          `${label} for ${horizon} must be a valid numeric value.`
        );
      }

      const number = Number(value);

      if (number < 0 || number > 100) {
        throw new Error(
          `${label} for ${horizon} must be between 0 and 100.`
        );
      }
    });
  });

  return true;
};

// --------------------------------------------------
// HEADER METRIC
// --------------------------------------------------

function HeaderMetric({ icon: Icon, label, value, description }) {
  return (
    <div className="rounded-2xl border border-white/10 bg-white/[0.07] p-5 backdrop-blur-sm transition hover:bg-white/[0.11]">
      <div className="flex items-center gap-3">
        <div className="rounded-xl bg-white/10 p-3 text-blue-200">
          <Icon size={22} />
        </div>

        <div className="min-w-0">
          <p className="text-sm text-blue-100/70">{label}</p>
          <h3 className="mt-1 text-2xl font-bold capitalize text-white">
            {value}
          </h3>
        </div>
      </div>

      <p className="mt-4 text-xs text-blue-100/60">
        {description}
      </p>
    </div>
  );
}

// --------------------------------------------------
// FORECAST CARD
// --------------------------------------------------

function ForecastCard({
  title,
  value,
  icon: Icon,
  description,
  accent,
  iconBackground,
}) {
  const hasValue =
    isFiniteNumber(value) &&
    Number(value) >= 0 &&
    Number(value) <= 100;

  const safeValue = hasValue ? Number(value) : 0;

  const progressColor =
    accent === "text-blue-600"
      ? "bg-blue-500"
      : "bg-violet-500";

  return (
    <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm transition duration-300 hover:-translate-y-1 hover:shadow-lg">
      <div className="flex items-start justify-between gap-4">
        <div>
          <p className="text-sm font-medium text-slate-500">
            {title}
          </p>

          <div className="mt-3 flex items-baseline gap-1">
            <h2 className="text-4xl font-extrabold tracking-tight text-slate-900">
              {hasValue ? safeValue.toFixed(2) : "--"}
            </h2>

            {hasValue && (
              <span className="text-lg font-semibold text-slate-400">
                %
              </span>
            )}
          </div>
        </div>

        <div className={`rounded-2xl p-4 ${iconBackground}`}>
          <Icon size={27} />
        </div>
      </div>

      <p className="mt-4 text-sm text-slate-500">
        {description}
      </p>

      <div className="mt-5">
        <div className="mb-2 flex items-center justify-between text-xs">
          <span className="font-medium text-slate-400">
            Utilization level
          </span>

          <span className={`font-bold ${accent}`}>
            {hasValue
              ? `${safeValue.toFixed(1)}%`
              : "Invalid or awaiting forecast"}
          </span>
        </div>

        <div className="h-2.5 overflow-hidden rounded-full bg-slate-100">
          <div
            className={`h-full rounded-full transition-all duration-700 ${progressColor}`}
            style={{ width: `${safeValue}%` }}
          />
        </div>
      </div>

      {hasValue && (
        <div className="mt-4 flex items-center gap-2 text-xs text-slate-400">
          <Activity size={14} />
          Predicted resource utilization
        </div>
      )}
    </div>
  );
}

// --------------------------------------------------
// MAIN FORECASTING PAGE
// --------------------------------------------------

function Forecasting() {
  const [health, setHealth] = useState(null);

  const [historyData, setHistoryData] = useState(null);

  const [machineId, setMachineId] = useState("1");

  const [forecast, setForecast] = useState(null);

  const [selectedHorizon, setSelectedHorizon] =
    useState("15min");

  const [healthLoading, setHealthLoading] = useState(true);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [loading, setLoading] = useState(false);

  const [historyError, setHistoryError] = useState("");
  const [error, setError] = useState("");

  // ------------------------------------------------
  // CHECK FORECASTING SERVICE
  // ------------------------------------------------

  const loadHealth = async () => {
    setHealthLoading(true);

    try {
      const data = await checkForecastHealth();

      if (!data || typeof data !== "object") {
        throw new Error("Invalid forecasting service response.");
      }

      setHealth(data);
    } catch {
      setHealth(null);
    } finally {
      setHealthLoading(false);
    }
  };

  // ------------------------------------------------
  // FETCH ACTUAL HISTORICAL DATA
  // ------------------------------------------------

  const loadHistoricalData = async (requestedMachineId = machineId) => {
    const parsedMachineId = Number(requestedMachineId);

    if (
      !Number.isInteger(parsedMachineId) ||
      parsedMachineId < 1
    ) {
      setHistoryError("Please enter a valid machine ID.");
      return;
    }

    setHistoryLoading(true);
    setHistoryError("");
    setError("");
    setForecast(null);
    setHistoryData(null);

    try {
      const result = await getHistoricalData(parsedMachineId);

      if (
        !result ||
        result.status !== "success" ||
        !Array.isArray(result.history)
      ) {
        throw new Error(
          "The historical data API returned an invalid response."
        );
      }

      validateHistoricalData(result.history);

      setHistoryData(result);
      setMachineId(String(parsedMachineId));
    } catch (err) {
      setHistoryData(null);

      setHistoryError(
        err.response?.data?.detail ||
          err.message ||
          "Unable to fetch historical observations."
      );
    } finally {
      setHistoryLoading(false);
    }
  };

  // ------------------------------------------------
  // LOAD SERVICE AND HISTORICAL DATA ON PAGE OPEN
  // ------------------------------------------------

  useEffect(() => {
    loadHealth();
    loadHistoricalData(1);
  }, []);

  // ------------------------------------------------
  // GENERATE FORECAST USING REAL HISTORY
  // ------------------------------------------------

  const handleGenerateForecast = async () => {
    if (!historyData?.history) {
      setError(
        "Please load historical data before generating a forecast."
      );
      return;
    }

    setLoading(true);
    setError("");
    setForecast(null);

    try {
      // Validate historical observations again before sending
      // them to the forecasting API.
      validateHistoricalData(historyData.history);

      const result = await generateForecast(historyData.history);

      // Validate both forecast horizons before displaying results.
      validateForecastResponse(result);

      setForecast(result);
    } catch (err) {
      setForecast(null);

      setError(
        err.response?.data?.detail ||
          err.message ||
          "Unable to generate forecast. Please try again."
      );
    } finally {
      setLoading(false);
    }
  };

  // ------------------------------------------------
  // SELECTED FORECAST
  // ------------------------------------------------

  const selectedForecast =
    forecast?.forecasts?.[selectedHorizon];

  const cpuValue = selectedForecast?.cpu_utilization;
  const memoryValue = selectedForecast?.memory_utilization;

  const serviceConnected = Boolean(health);
  const historyAvailable = Boolean(historyData);

  // ------------------------------------------------
  // UI
  // ------------------------------------------------

  return (
    <div className="min-h-screen bg-slate-100">

      {/* ==========================================
          DARK HERO SECTION
      =========================================== */}

      <section className="relative overflow-hidden bg-gradient-to-br from-[#080f25] via-[#172653] to-[#32106b] px-6 pb-10 pt-9 md:px-9 lg:px-10">

        <div className="pointer-events-none absolute -right-20 -top-32 h-96 w-96 rounded-full bg-blue-500/20 blur-3xl" />

        <div className="pointer-events-none absolute bottom-0 left-1/3 h-64 w-64 rounded-full bg-violet-500/10 blur-3xl" />

        <div className="relative z-10">

          {/* TOP LABEL */}

          <div className="mb-7 flex flex-wrap items-center gap-4">

            <div className="inline-flex items-center gap-2 rounded-full border border-violet-400/20 bg-violet-500/10 px-4 py-2 text-xs font-bold tracking-wide text-violet-100">
              <Activity size={16} />
              CLOUD INTELLIGENCE
            </div>

            <div className="flex items-center gap-2 text-sm font-semibold">

              <span
                className={`h-2.5 w-2.5 rounded-full ${
                  healthLoading
                    ? "animate-pulse bg-amber-400"
                    : serviceConnected
                    ? "bg-emerald-400"
                    : "bg-red-400"
                }`}
              />

              <span
                className={
                  healthLoading
                    ? "text-amber-300"
                    : serviceConnected
                    ? "text-emerald-300"
                    : "text-red-300"
                }
              >
                {healthLoading
                  ? "Checking Forecasting Engine"
                  : serviceConnected
                  ? "Forecasting Engine Active"
                  : "Forecasting Engine Offline"}
              </span>
            </div>

          </div>

          {/* MAIN HEADER */}

          <div className="flex flex-col justify-between gap-7 xl:flex-row xl:items-center">

            <div className="max-w-3xl">

              <h1 className="text-4xl font-extrabold tracking-tight text-white md:text-5xl">
                Resource Forecasting
              </h1>

              <p className="mt-4 max-w-3xl text-base leading-7 text-blue-100/80 md:text-lg">
                Predict future CPU and memory utilization using
                historical infrastructure data from CloudMorph's
                forecasting pipeline.
              </p>

              <div className="mt-6 flex flex-wrap items-center gap-x-6 gap-y-3 text-sm text-blue-100/70">

                <span className="flex items-center gap-2">
                  <Database size={15} />
                  Alibaba historical dataset
                </span>

                <span className="flex items-center gap-2">
                  <Clock size={15} />
                  15 & 30 minute predictions
                </span>

                <span className="flex items-center gap-2">
                  <BrainCircuit size={15} />
                  Time-series inference
                </span>

              </div>
            </div>

            {/* GENERATE BUTTON */}

            <div className="shrink-0">

              <button
                onClick={handleGenerateForecast}
                disabled={
                  loading ||
                  historyLoading ||
                  !historyAvailable
                }
                className="flex w-full items-center justify-center gap-3 rounded-xl bg-gradient-to-r from-blue-500 to-violet-600 px-7 py-4 font-bold text-white shadow-lg shadow-blue-950/30 transition hover:scale-[1.02] hover:shadow-xl disabled:cursor-not-allowed disabled:opacity-50 sm:w-auto"
              >
                <RefreshCw
                  size={19}
                  className={loading ? "animate-spin" : ""}
                />

                {loading
                  ? "Generating Forecast..."
                  : "Generate Forecast"}
              </button>

              {!serviceConnected && !healthLoading && (
                <button
                  onClick={loadHealth}
                  className="mt-3 w-full text-center text-xs font-semibold text-blue-200 underline underline-offset-4"
                >
                  Retry service connection
                </button>
              )}

            </div>

          </div>

          {/* HEADER METRICS */}

          <div className="mt-10 grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">

            <HeaderMetric
              icon={Database}
              label="Data Source"
              value="Alibaba Trace"
              description="Historical server resource records"
            />

            <HeaderMetric
              icon={Server}
              label="Selected Machine"
              value={`Machine ${machineId}`}
              description="Machine selected for forecasting"
            />

            <HeaderMetric
              icon={Layers}
              label="Input Window"
              value="12 Records"
              description="Historical observations per prediction"
            />

            <HeaderMetric
              icon={BrainCircuit}
              label="Model Status"
              value={health?.model_status || "Baseline"}
              description="Current forecasting method"
            />

          </div>

        </div>
      </section>

      {/* ==========================================
          MAIN CONTENT
      =========================================== */}

      <main className="px-5 py-8 md:px-8 lg:px-10">

        {/* HISTORICAL DATA SOURCE */}

        <section className="mb-6 rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">

          <div className="flex flex-col justify-between gap-5 lg:flex-row lg:items-center">

            <div className="flex items-center gap-4">

              <div className="rounded-xl bg-blue-50 p-3 text-blue-600">
                <Database size={25} />
              </div>

              <div>
                <h2 className="text-base font-bold text-slate-900">
                  Historical Resource Data
                </h2>

                <p className="mt-1 text-sm text-slate-500">
                  Fetch actual CPU and memory observations from the Alibaba dataset.
                </p>
              </div>

            </div>

            <div className="flex flex-col gap-3 sm:flex-row">

              <div className="flex items-center gap-2 rounded-xl border border-slate-200 px-3 py-2">

                <Server size={17} className="text-slate-400" />

                <input
                  type="number"
                  min="1"
                  value={machineId}
                  onChange={(e) => setMachineId(e.target.value)}
                  placeholder="Machine ID"
                  className="w-28 bg-transparent text-sm font-semibold text-slate-800 outline-none"
                />

              </div>

              <button
                onClick={() => loadHistoricalData(machineId)}
                disabled={historyLoading || loading}
                className="flex items-center justify-center gap-2 rounded-xl bg-blue-600 px-5 py-3 text-sm font-semibold text-white transition hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-50"
              >
                {historyLoading ? (
                  <RefreshCw size={17} className="animate-spin" />
                ) : (
                  <Search size={17} />
                )}

                {historyLoading ? "Loading..." : "Load Data"}
              </button>

            </div>

          </div>

          {/* HISTORICAL DATA STATUS */}

          <div className="mt-5 border-t border-slate-100 pt-4">

            {historyLoading ? (
              <div className="flex items-center gap-2 text-sm text-blue-600">
                <RefreshCw size={16} className="animate-spin" />
                Fetching historical observations...
              </div>
            ) : historyAvailable ? (
              <div className="flex flex-wrap items-center gap-x-6 gap-y-3 text-sm">

                <span className="flex items-center gap-2 font-semibold text-emerald-600">
                  <CheckCircle2 size={17} />
                  Historical data loaded
                </span>

                <span className="text-slate-500">
                  Machine ID:{" "}
                  <strong className="text-slate-800">
                    {historyData.machine_id}
                  </strong>
                </span>

                <span className="text-slate-500">
                  Observations:{" "}
                  <strong className="text-slate-800">
                    {historyData.observations}
                  </strong>
                </span>

                <span className="text-slate-500">
                  Source:{" "}
                  <strong className="text-slate-800">
                    {historyData.source}
                  </strong>
                </span>

              </div>
            ) : (
              <div className="flex items-center gap-2 text-sm text-slate-500">
                <AlertCircle size={17} />
                Historical data has not been loaded.
              </div>
            )}

          </div>

        </section>

        {/* SERVICE INFORMATION */}

        <section className="mb-6 rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">

          <div className="flex flex-col justify-between gap-4 sm:flex-row sm:items-center">

            <div className="flex items-center gap-4">

              <div className="rounded-xl bg-blue-50 p-3 text-blue-600">
                <BrainCircuit size={25} />
              </div>

              <div>
                <h2 className="text-base font-bold text-slate-900">
                  Forecasting Engine
                </h2>

                <p className="mt-1 text-sm text-slate-500">
                  FastAPI resource utilization inference service
                </p>
              </div>

            </div>

            <div className="flex items-center gap-2">

              {healthLoading ? (
                <>
                  <RefreshCw size={17} className="animate-spin text-blue-500" />
                  <span className="text-sm font-semibold text-slate-500">
                    Checking service...
                  </span>
                </>
              ) : serviceConnected ? (
                <>
                  <CheckCircle2 size={18} className="text-emerald-500" />
                  <span className="text-sm font-semibold text-emerald-600">
                    Service Connected
                  </span>
                </>
              ) : (
                <>
                  <AlertCircle size={18} className="text-red-500" />
                  <span className="text-sm font-semibold text-red-600">
                    Service Unavailable
                  </span>
                </>
              )}

            </div>

          </div>

          {health && (
            <div className="mt-5 flex flex-wrap gap-2 border-t border-slate-100 pt-4">

              <span className="rounded-full bg-blue-50 px-3 py-1.5 text-xs font-semibold text-blue-700">
                Method: {health.forecast_method}
              </span>

              <span className="rounded-full bg-amber-50 px-3 py-1.5 text-xs font-semibold text-amber-700">
                Model: {health.model_status}
              </span>

              <span className="rounded-full bg-violet-50 px-3 py-1.5 text-xs font-semibold text-violet-700">
                15 & 30 Minute Horizons
              </span>

            </div>
          )}

        </section>

        {/* FORECAST HORIZON */}

        <section className="mb-7">

          <div className="mb-4 flex flex-col justify-between gap-3 sm:flex-row sm:items-center">

            <div>
              <div className="flex items-center gap-2">

                <Clock size={19} className="text-blue-600" />

                <h2 className="text-xl font-bold text-slate-900">
                  Forecast Overview
                </h2>

              </div>

              <p className="mt-1 text-sm text-slate-500">
                Select the prediction time horizon.
              </p>
            </div>

            <div className="inline-flex w-fit rounded-xl border border-slate-200 bg-white p-1.5 shadow-sm">

              {[
                { label: "15 Minutes", value: "15min" },
                { label: "30 Minutes", value: "30min" },
              ].map((item) => (
                <button
                  key={item.value}
                  onClick={() => setSelectedHorizon(item.value)}
                  className={`rounded-lg px-5 py-2.5 text-sm font-semibold transition ${
                    selectedHorizon === item.value
                      ? "bg-gradient-to-r from-blue-600 to-violet-600 text-white shadow-md"
                      : "text-slate-500 hover:bg-slate-50 hover:text-slate-800"
                  }`}
                >
                  {item.label}
                </button>
              ))}

            </div>

          </div>

        </section>

        {/* CPU AND MEMORY CARDS */}

        <section className="mb-7 grid grid-cols-1 gap-5 lg:grid-cols-2">

          <ForecastCard
            title="Predicted CPU Utilization"
            value={cpuValue}
            icon={Cpu}
            description="Expected processor utilization for the selected forecast horizon."
            accent="text-blue-600"
            iconBackground="bg-blue-50 text-blue-600"
          />

          <ForecastCard
            title="Predicted Memory Utilization"
            value={memoryValue}
            icon={MemoryStick}
            description="Expected memory consumption for the selected forecast horizon."
            accent="text-violet-600"
            iconBackground="bg-violet-50 text-violet-600"
          />

        </section>

        {/* HISTORICAL DATA ERROR */}

        {historyError && (
          <section className="mb-7 flex gap-3 rounded-2xl border border-red-200 bg-red-50 p-5">

            <AlertCircle size={21} className="mt-0.5 shrink-0 text-red-600" />

            <div>
              <h3 className="font-bold text-red-800">
                Historical Data Loading Failed
              </h3>

              <p className="mt-1 text-sm text-red-700">
                {historyError}
              </p>
            </div>

          </section>
        )}

        {/* FORECAST ERROR */}

        {error && (
          <section className="mb-7 flex gap-3 rounded-2xl border border-red-200 bg-red-50 p-5">

            <AlertCircle size={21} className="mt-0.5 shrink-0 text-red-600" />

            <div>
              <h3 className="font-bold text-red-800">
                Forecast Generation Failed
              </h3>

              <p className="mt-1 text-sm text-red-700">
                {error}
              </p>
            </div>

          </section>
        )}

        {/* EMPTY STATE */}

        {!forecast && !loading && !error && (
          <section className="mb-7 rounded-2xl border border-dashed border-slate-300 bg-white px-6 py-12 text-center shadow-sm">

            <div className="mx-auto mb-5 flex h-16 w-16 items-center justify-center rounded-2xl bg-gradient-to-br from-blue-50 to-violet-50 text-blue-600">
              <TrendingUp size={30} />
            </div>

            <h3 className="text-xl font-bold text-slate-900">
              {historyAvailable
                ? "Historical data is ready"
                : "Waiting for historical data"}
            </h3>

            <p className="mx-auto mt-2 max-w-lg text-sm leading-6 text-slate-500">
              {historyAvailable
                ? `12 actual historical observations for machine ${historyData.machine_id} have been loaded. Generate a forecast to view predicted CPU and memory utilization.`
                : "Load historical observations before generating a resource forecast."}
            </p>

            <button
              onClick={handleGenerateForecast}
              disabled={!historyAvailable || loading}
              className="mt-6 inline-flex items-center gap-2 rounded-xl bg-gradient-to-r from-blue-600 to-violet-600 px-6 py-3 font-semibold text-white shadow-md transition hover:shadow-lg disabled:cursor-not-allowed disabled:opacity-50"
            >
              <Zap size={17} />
              Generate Forecast
            </button>

          </section>
        )}

        {/* LOADING STATE */}

        {loading && (
          <section className="mb-7 rounded-2xl border border-slate-200 bg-white px-6 py-12 text-center shadow-sm">

            <RefreshCw
              size={32}
              className="mx-auto animate-spin text-blue-600"
            />

            <h3 className="mt-4 text-lg font-bold text-slate-900">
              Generating your forecast
            </h3>

            <p className="mt-2 text-sm text-slate-500">
              Processing 12 historical observations for machine {machineId}...
            </p>

          </section>
        )}

        {/* SUCCESS INFORMATION */}

        {forecast && (
          <section className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">

            <div className="mb-6 flex flex-col justify-between gap-4 sm:flex-row sm:items-center">

              <div className="flex items-center gap-3">

                <div className="rounded-xl bg-emerald-50 p-3 text-emerald-600">
                  <CheckCircle2 size={24} />
                </div>

                <div>
                  <h3 className="text-lg font-bold text-slate-900">
                    Forecast Generated Successfully
                  </h3>

                  <p className="mt-1 text-sm text-slate-500">
                    Results generated using historical dataset observations.
                  </p>
                </div>

              </div>

              <span className="w-fit rounded-full bg-emerald-50 px-3 py-1.5 text-xs font-bold text-emerald-700">
                Successful
              </span>

            </div>

            <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">

              <div className="rounded-xl bg-slate-50 p-5">
                <div className="flex items-center gap-2 text-slate-400">
                  <Gauge size={16} />
                  <p className="text-xs font-semibold uppercase tracking-wide">
                    Forecast Method
                  </p>
                </div>

                <p className="mt-3 text-lg font-bold text-slate-800">
                  {forecast.forecast_method}
                </p>
              </div>

              <div className="rounded-xl bg-slate-50 p-5">
                <div className="flex items-center gap-2 text-slate-400">
                  <Layers size={16} />
                  <p className="text-xs font-semibold uppercase tracking-wide">
                    Input Window
                  </p>
                </div>

                <p className="mt-3 text-lg font-bold text-slate-800">
                  {forecast.input_window} observations
                </p>
              </div>

              <div className="rounded-xl bg-slate-50 p-5">
                <div className="flex items-center gap-2 text-slate-400">
                  <BrainCircuit size={16} />
                  <p className="text-xs font-semibold uppercase tracking-wide">
                    Model Status
                  </p>
                </div>

                <p className="mt-3 text-lg font-bold capitalize text-amber-600">
                  {forecast.model_status}
                </p>
              </div>

            </div>

            <div className="mt-5 flex flex-wrap items-center gap-4 border-t border-slate-100 pt-4 text-xs text-slate-500">

              <span className="flex items-center gap-2">
                <Database size={15} />
                {historyData?.source}
              </span>

              <span className="flex items-center gap-2">
                <Server size={15} />
                Machine {historyData?.machine_id}
              </span>

              <span className="flex items-center gap-2">
                <Activity size={15} />
                Six input features
              </span>

            </div>

          </section>
        )}

        {/* FOOTER */}

        <div className="mt-8 flex flex-col gap-2 border-t border-slate-200 pt-5 text-xs text-slate-400 sm:flex-row sm:items-center sm:justify-between">

          <span>CloudMorph Resource Forecasting</span>

          <span>
            Historical data source: Alibaba Cluster Trace 2017
          </span>

        </div>

      </main>
    </div>
  );
}

export default Forecasting;