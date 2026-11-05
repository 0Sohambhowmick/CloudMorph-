import { useState } from "react";
import {
  Activity,
  Bell,
  BrainCircuit,
  Check,
  ChevronDown,
  Cloud,
  DollarSign,
  Globe2,
  KeyRound,
  Lock,
  Mail,
  RefreshCw,
  Save,
  Server,
  ShieldCheck,
  SlidersHorizontal,
  Sparkles,
  UserRound,
  Zap,
} from "lucide-react";

const SETTINGS_STORAGE_KEY = "cloudmorph_settings";

function getSavedSettings() {
  try {
    if (typeof window === "undefined") return {};
    const saved = localStorage.getItem(SETTINGS_STORAGE_KEY);
    return saved ? JSON.parse(saved) : {};
  } catch {
    return {};
  }
}

function Settings() {
  /* ========================================================= */
  /* STATE */
  /* ========================================================= */

  const [workspaceName, setWorkspaceName] =
    useState(getSavedSettings().workspaceName ?? "CloudMorph Workspace");

  const [environment, setEnvironment] =
    useState(getSavedSettings().environment ?? "Production");

  const [region, setRegion] =
    useState(getSavedSettings().region ?? "ap-south-1");

  const [currency, setCurrency] =
    useState(getSavedSettings().currency ?? "USD");

  const [monitoringFrequency, setMonitoringFrequency] =
    useState(getSavedSettings().monitoringFrequency ?? "5 minutes");

  const [autoRefresh, setAutoRefresh] =
    useState(getSavedSettings().autoRefresh ?? true);

  const [resourceMonitoring, setResourceMonitoring] =
    useState(getSavedSettings().resourceMonitoring ?? true);

  const [costMonitoring, setCostMonitoring] =
    useState(getSavedSettings().costMonitoring ?? true);

  const [aiRecommendations, setAiRecommendations] =
    useState(getSavedSettings().aiRecommendations ?? true);

  const [optimizationMode, setOptimizationMode] =
    useState(getSavedSettings().optimizationMode ?? "Balanced");

  const [costPriority, setCostPriority] =
    useState(getSavedSettings().costPriority ?? 80);

  const [performancePriority, setPerformancePriority] =
    useState(getSavedSettings().performancePriority ?? 75);

  const [riskTolerance, setRiskTolerance] =
    useState(getSavedSettings().riskTolerance ?? "Medium");

  const [costAlerts, setCostAlerts] =
    useState(getSavedSettings().costAlerts ?? true);

  const [infrastructureAlerts, setInfrastructureAlerts] =
    useState(getSavedSettings().infrastructureAlerts ?? true);

  const [aiAlerts, setAiAlerts] =
    useState(getSavedSettings().aiAlerts ?? true);

  const [emailNotifications, setEmailNotifications] =
    useState(getSavedSettings().emailNotifications ?? false);

  const [isSaving, setIsSaving] =
    useState(false);

  const [saved, setSaved] =
    useState(false);

  const [saveError, setSaveError] = useState("");

  /* ========================================================= */
  /* SAVE SETTINGS */
  /* ========================================================= */

  const handleSave = async () => {
    setIsSaving(true);
    setSaved(false);
    setSaveError("");

    const settings = {
      workspaceName: workspaceName.trim() || "CloudMorph Workspace",
      environment,
      region,
      currency,
      monitoringFrequency,
      autoRefresh,
      resourceMonitoring,
      costMonitoring,
      aiRecommendations,
      optimizationMode,
      costPriority: Number(costPriority),
      performancePriority: Number(performancePriority),
      riskTolerance,
      costAlerts,
      infrastructureAlerts,
      aiAlerts,
      emailNotifications,
    };

    try {
      // These preferences are stored locally because a dedicated settings
      // endpoint is not confirmed in the current backend.
      localStorage.setItem(SETTINGS_STORAGE_KEY, JSON.stringify(settings));

      // Keep the workspace name consistent with the Profile page.
      localStorage.setItem("cloudmorph_workspace", settings.workspaceName);

      setSaved(true);
      window.setTimeout(() => setSaved(false), 3000);
    } catch (error) {
      console.error("Could not save CloudMorph settings:", error);
      setSaveError(
        "Unable to save settings on this device. Check browser storage permissions and try again."
      );
    } finally {
      setIsSaving(false);
    }
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
                  <SlidersHorizontal size={15} />
                  System Configuration
                </span>

                <span className="flex items-center gap-2 text-xs font-semibold text-emerald-300">
                  <span className="h-2 w-2 animate-pulse rounded-full bg-emerald-400" />
                  Configuration Active
                </span>
              </div>

              <h1 className="mt-6 text-3xl font-bold tracking-tight text-white md:text-5xl">
                Settings
              </h1>

              <p className="mt-4 max-w-2xl text-sm leading-relaxed text-slate-300 md:text-base">
                Configure your CloudMorph workspace, monitoring
                behavior, AI optimization preferences, notifications,
                and security settings.
              </p>
            </div>

            {/* SAVE BUTTON */}
            <button
              type="button"
              onClick={handleSave}
              disabled={isSaving}
              className="flex items-center justify-center gap-2 rounded-xl bg-white px-5 py-3.5 text-sm font-bold text-slate-900 shadow-lg transition hover:bg-slate-100 disabled:cursor-not-allowed disabled:opacity-70"
            >
              {isSaving ? (
                <RefreshCw
                  size={18}
                  className="animate-spin"
                />
              ) : saved ? (
                <Check size={18} />
              ) : (
                <Save size={18} />
              )}

              {isSaving
                ? "Saving..."
                : saved
                ? "Changes Saved"
                : "Save Changes"}
            </button>
          </div>

          {/* HERO STATUS */}
          <div className="mt-8 flex flex-wrap items-center gap-4">
            <StatusBadge
              icon={Activity}
              label="Monitoring Active"
            />

            <StatusBadge
              icon={BrainCircuit}
              label="AI Engine Active"
            />

            <StatusBadge
              icon={ShieldCheck}
              label="Security Enabled"
            />
          </div>
        </div>
      </section>

      {/* ===================================================== */}
      {/* MAIN */}
      {/* ===================================================== */}

      <main className="p-8">
        <div className="grid grid-cols-1 gap-8 xl:grid-cols-3">
          {/* ================================================= */}
          {/* LEFT / MAIN SETTINGS */}
          {/* ================================================= */}

          <div className="space-y-8 xl:col-span-2">
            {/* ================================================= */}
            {/* GENERAL */}
            {/* ================================================= */}

            <SettingsSection
              icon={Globe2}
              title="General"
              description="Configure the basic settings for your CloudMorph workspace."
            >
              <div className="grid grid-cols-1 gap-5 md:grid-cols-2">
                <InputField
                  label="Workspace Name"
                  value={workspaceName}
                  onChange={setWorkspaceName}
                  icon={Cloud}
                />

                <SelectField
                  label="Environment"
                  value={environment}
                  onChange={setEnvironment}
                  icon={Server}
                  options={[
                    "Production",
                    "Staging",
                    "Development",
                  ]}
                />

                <SelectField
                  label="Default Region"
                  value={region}
                  onChange={setRegion}
                  icon={Globe2}
                  options={[
                    "ap-south-1",
                    "us-east-1",
                    "us-west-2",
                    "eu-west-1",
                    "asia-south1",
                  ]}
                />

                <SelectField
                  label="Default Currency"
                  value={currency}
                  onChange={setCurrency}
                  icon={DollarSign}
                  options={[
                    "USD",
                    "INR",
                    "EUR",
                    "GBP",
                  ]}
                />
              </div>
            </SettingsSection>

            {/* ================================================= */}
            {/* MONITORING */}
            {/* ================================================= */}

            <SettingsSection
              icon={Activity}
              title="Monitoring"
              description="Control how CloudMorph monitors your cloud infrastructure."
            >
              <div className="space-y-4">
                <SelectField
                  label="Monitoring Frequency"
                  value={monitoringFrequency}
                  onChange={setMonitoringFrequency}
                  icon={RefreshCw}
                  options={[
                    "1 minute",
                    "5 minutes",
                    "15 minutes",
                    "30 minutes",
                    "1 hour",
                  ]}
                />

                <ToggleRow
                  icon={RefreshCw}
                  title="Automatic Refresh"
                  description="Set your preferred refresh behavior; automatic telemetry refresh requires monitoring integration."
                  enabled={autoRefresh}
                  onChange={setAutoRefresh}
                />

                <ToggleRow
                  icon={Server}
                  title="Resource Monitoring"
                  description="Set your resource-monitoring preference for future monitoring integration."
                  enabled={resourceMonitoring}
                  onChange={setResourceMonitoring}
                />

                <ToggleRow
                  icon={DollarSign}
                  title="Cost Monitoring"
                  description="Set your cost-monitoring preference for future cost integration."
                  enabled={costMonitoring}
                  onChange={setCostMonitoring}
                />
              </div>
            </SettingsSection>

            {/* ================================================= */}
            {/* AI OPTIMIZATION */}
            {/* ================================================= */}

            <SettingsSection
              icon={BrainCircuit}
              title="AI Optimization"
              description="Configure how CloudMorph evaluates optimization opportunities."
            >
              <div className="space-y-5">
                <ToggleRow
                  icon={Sparkles}
                  title="AI Recommendations"
                  description="Allow CloudMorph to generate KEEP, RESIZE, and SWITCH recommendations."
                  enabled={aiRecommendations}
                  onChange={setAiRecommendations}
                />

                <SelectField
                  label="Optimization Mode"
                  value={optimizationMode}
                  onChange={setOptimizationMode}
                  icon={BrainCircuit}
                  options={[
                    "Balanced",
                    "Cost Focused",
                    "Performance Focused",
                    "Conservative",
                  ]}
                />

                <RangeField
                  label="Cost Priority"
                  value={costPriority}
                  onChange={setCostPriority}
                  description="How strongly cost efficiency should influence AI decisions."
                />

                <RangeField
                  label="Performance Priority"
                  value={performancePriority}
                  onChange={setPerformancePriority}
                  description="How strongly application performance should influence AI decisions."
                />

                <SelectField
                  label="Risk Tolerance"
                  value={riskTolerance}
                  onChange={setRiskTolerance}
                  icon={ShieldCheck}
                  options={[
                    "Low",
                    "Medium",
                    "High",
                  ]}
                />
              </div>
            </SettingsSection>

            {/* ================================================= */}
            {/* NOTIFICATIONS */}
            {/* ================================================= */}

            <SettingsSection
              icon={Bell}
              title="Notifications"
              description="Choose which CloudMorph events should generate notifications."
            >
              <div className="space-y-4">
                <ToggleRow
                  icon={DollarSign}
                  title="Cost Alerts"
                  description="Receive alerts when cloud spending exceeds configured thresholds."
                  enabled={costAlerts}
                  onChange={setCostAlerts}
                />

                <ToggleRow
                  icon={Server}
                  title="Infrastructure Alerts"
                  description="Receive alerts when infrastructure health or utilization requires attention."
                  enabled={infrastructureAlerts}
                  onChange={setInfrastructureAlerts}
                />

                <ToggleRow
                  icon={Sparkles}
                  title="AI Recommendation Alerts"
                  description="Receive notifications when CloudMorph generates important optimization recommendations."
                  enabled={aiAlerts}
                  onChange={setAiAlerts}
                />

                <ToggleRow
                  icon={Mail}
                  title="Email Notifications"
                  description="Save your email-notification preference; email delivery requires notification-service integration."
                  enabled={emailNotifications}
                  onChange={setEmailNotifications}
                />
              </div>
            </SettingsSection>

            {/* ================================================= */}
            {/* SECURITY */}
            {/* ================================================= */}

            <SettingsSection
              icon={Lock}
              title="Security"
              description="Review the security and cloud connection configuration."
            >
              <div className="space-y-4">
                <SecurityRow
                  icon={KeyRound}
                  title="Cloud Provider Credentials"
                  description="Provider credentials are securely configured."
                  status="Connected"
                />

                <SecurityRow
                  icon={ShieldCheck}
                  title="Authentication"
                  description="Workspace authentication and access control are enabled."
                  status="Enabled"
                />

                <SecurityRow
                  icon={Lock}
                  title="Data Protection"
                  description="Application and infrastructure configuration data is protected."
                  status="Protected"
                />
              </div>
            </SettingsSection>
          </div>

          {/* ================================================= */}
          {/* RIGHT SIDEBAR */}
          {/* ================================================= */}

          <aside className="space-y-8">
            {/* WORKSPACE STATUS */}
            <div className="overflow-hidden rounded-3xl border border-slate-200 bg-white shadow-sm">
              <div className="bg-gradient-to-br from-blue-600 to-violet-600 p-6 text-white">
                <div className="flex items-center gap-3">
                  <div className="rounded-xl bg-white/15 p-3">
                    <Cloud size={22} />
                  </div>

                  <div>
                    <p className="text-xs font-medium text-blue-100">
                      Workspace
                    </p>

                    <h3 className="mt-1 text-lg font-bold">
                      CloudMorph
                    </h3>
                  </div>
                </div>

                <div className="mt-6 rounded-2xl bg-white/10 p-4">
                  <div className="flex items-center gap-2">
                    <span className="h-2.5 w-2.5 animate-pulse rounded-full bg-emerald-300" />

                    <span className="text-sm font-bold">
                      All Systems Operational
                    </span>
                  </div>

                  <p className="mt-2 text-xs leading-relaxed text-blue-100">
                    Monitoring, analytics, AI optimization, and
                    infrastructure services are active.
                  </p>
                </div>
              </div>

              <div className="p-6">
                <StatusItem
                  label="Infrastructure"
                  status="Active"
                />

                <StatusItem
                  label="Analytics Engine"
                  status="Active"
                />

                <StatusItem
                  label="AI Optimizer"
                  status="Active"
                />

                <StatusItem
                  label="Notifications"
                  status={
                    emailNotifications
                      ? "Active"
                      : "In-App Only"
                  }
                  last
                />
              </div>
            </div>

            {/* OPTIMIZATION PROFILE */}
            <div className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm">
              <div className="flex items-center gap-3">
                <div className="rounded-xl bg-violet-50 p-3 text-violet-600">
                  <Zap size={21} />
                </div>

                <div>
                  <h3 className="font-bold text-slate-800">
                    Optimization Profile
                  </h3>

                  <p className="text-xs text-slate-400">
                    Current AI configuration
                  </p>
                </div>
              </div>

              <div className="mt-6 space-y-4">
                <ProfileMetric
                  label="Cost Priority"
                  value={costPriority}
                />

                <ProfileMetric
                  label="Performance"
                  value={performancePriority}
                />

                <div className="rounded-xl bg-slate-50 p-4">
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-medium text-slate-400">
                      Risk Tolerance
                    </span>

                    <span className="text-xs font-bold text-violet-600">
                      {riskTolerance}
                    </span>
                  </div>
                </div>

                <div className="rounded-xl bg-slate-50 p-4">
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-medium text-slate-400">
                      Mode
                    </span>

                    <span className="text-xs font-bold text-blue-600">
                      {optimizationMode}
                    </span>
                  </div>
                </div>
              </div>
            </div>

            {/* ACCOUNT */}
            <div className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm">
              <div className="flex items-center gap-3">
                <div className="rounded-xl bg-blue-50 p-3 text-blue-600">
                  <UserRound size={21} />
                </div>

                <div>
                  <h3 className="font-bold text-slate-800">
                    Workspace Owner
                  </h3>

                  <p className="text-xs text-slate-400">
                    Administrator access
                  </p>
                </div>
              </div>

              <div className="mt-5 rounded-xl bg-slate-50 p-4">
                <p className="text-xs font-bold text-slate-700">
                  CloudMorph Administrator
                </p>

                <p className="mt-1 text-xs text-slate-400">
                  Full workspace configuration access
                </p>
              </div>
            </div>
          </aside>
        </div>

        {/* ===================================================== */}
        {/* BOTTOM SAVE BAR */}
        {/* ===================================================== */}

        {saveError && (
          <div
            role="status"
            className="mb-4 rounded-xl border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-700"
          >
            {saveError}
          </div>
        )}

        <div className="sticky bottom-4 z-20 mt-8">
          <div className="flex flex-col gap-4 rounded-2xl border border-slate-200 bg-white/95 p-4 shadow-xl backdrop-blur sm:flex-row sm:items-center sm:justify-between">
            <div className="flex items-center gap-3">
              <div
                className={`rounded-xl p-2.5 ${
                  saved
                    ? "bg-emerald-50 text-emerald-600"
                    : "bg-blue-50 text-blue-600"
                }`}
              >
                {saved ? (
                  <Check size={19} />
                ) : (
                  <Save size={19} />
                )}
              </div>

              <div>
                <p className="text-sm font-bold text-slate-700">
                  {saved
                    ? "Your settings have been saved."
                    : "Save your preferences to keep them after refresh."}
                </p>

                <p className="text-xs text-slate-400">
                  Preferences are stored in this browser and restored when you return on this device.
                </p>
              </div>
            </div>

            <button
              type="button"
              onClick={handleSave}
              disabled={isSaving}
              className="flex items-center justify-center gap-2 rounded-xl bg-slate-900 px-5 py-3 text-sm font-bold text-white transition hover:bg-blue-600 disabled:cursor-not-allowed disabled:opacity-70"
            >
              {isSaving ? (
                <RefreshCw
                  size={17}
                  className="animate-spin"
                />
              ) : saved ? (
                <Check size={17} />
              ) : (
                <Save size={17} />
              )}

              {isSaving
                ? "Saving..."
                : saved
                ? "Saved"
                : "Save Settings"}
            </button>
          </div>
        </div>

        <div className="h-6" />
      </main>
    </div>
  );
}

/* ========================================================= */
/* SETTINGS SECTION */
/* ========================================================= */

function SettingsSection({
  icon: Icon,
  title,
  description,
  children,
}) {
  return (
    <section className="rounded-3xl border border-slate-200 bg-white p-7 shadow-sm">
      <div className="flex items-start gap-4">
        <div className="rounded-xl bg-blue-50 p-3 text-blue-600">
          <Icon size={21} />
        </div>

        <div>
          <h2 className="text-xl font-bold text-slate-800">
            {title}
          </h2>

          <p className="mt-1 text-sm leading-relaxed text-slate-500">
            {description}
          </p>
        </div>
      </div>

      <div className="mt-7">
        {children}
      </div>
    </section>
  );
}

/* ========================================================= */
/* INPUT FIELD */
/* ========================================================= */

function InputField({
  label,
  value,
  onChange,
  icon: Icon,
}) {
  return (
    <div>
      <label className="text-xs font-bold text-slate-600">
        {label}
      </label>

      <div className="relative mt-2">
        <Icon
          size={17}
          className="absolute left-4 top-1/2 -translate-y-1/2 text-slate-400"
        />

        <input
          type="text"
          value={value}
          onChange={(event) =>
            onChange(event.target.value)
          }
          className="w-full rounded-xl border border-slate-200 bg-slate-50 py-3.5 pl-11 pr-4 text-sm font-medium text-slate-700 outline-none transition focus:border-blue-400 focus:bg-white focus:ring-4 focus:ring-blue-100"
        />
      </div>
    </div>
  );
}

/* ========================================================= */
/* SELECT FIELD */
/* ========================================================= */

function SelectField({
  label,
  value,
  onChange,
  icon: Icon,
  options,
}) {
  return (
    <div>
      <label className="text-xs font-bold text-slate-600">
        {label}
      </label>

      <div className="relative mt-2">
        <Icon
          size={17}
          className="pointer-events-none absolute left-4 top-1/2 -translate-y-1/2 text-slate-400"
        />

        <select
          value={value}
          onChange={(event) =>
            onChange(event.target.value)
          }
          className="w-full appearance-none rounded-xl border border-slate-200 bg-slate-50 py-3.5 pl-11 pr-11 text-sm font-medium text-slate-700 outline-none transition focus:border-blue-400 focus:bg-white focus:ring-4 focus:ring-blue-100"
        >
          {options.map((option) => (
            <option
              key={option}
              value={option}
            >
              {option}
            </option>
          ))}
        </select>

        <ChevronDown
          size={17}
          className="pointer-events-none absolute right-4 top-1/2 -translate-y-1/2 text-slate-400"
        />
      </div>
    </div>
  );
}

/* ========================================================= */
/* TOGGLE ROW */
/* ========================================================= */

function ToggleRow({
  icon: Icon,
  title,
  description,
  enabled,
  onChange,
}) {
  return (
    <div className="flex flex-col gap-4 rounded-2xl border border-slate-100 bg-slate-50 p-5 sm:flex-row sm:items-center sm:justify-between">
      <div className="flex items-start gap-4">
        <div className="rounded-xl bg-white p-3 text-blue-600 shadow-sm">
          <Icon size={19} />
        </div>

        <div>
          <p className="text-sm font-bold text-slate-700">
            {title}
          </p>

          <p className="mt-1 max-w-2xl text-xs leading-relaxed text-slate-400">
            {description}
          </p>
        </div>
      </div>

      <button
        type="button"
        onClick={() => onChange(!enabled)}
        className={`relative h-7 w-12 shrink-0 rounded-full transition ${
          enabled
            ? "bg-blue-600"
            : "bg-slate-300"
        }`}
        aria-label={`Toggle ${title}`}
      >
        <span
          className={`absolute top-1 h-5 w-5 rounded-full bg-white shadow-sm transition ${
            enabled
              ? "left-6"
              : "left-1"
          }`}
        />
      </button>
    </div>
  );
}

/* ========================================================= */
/* RANGE FIELD */
/* ========================================================= */

function RangeField({
  label,
  value,
  onChange,
  description,
}) {
  return (
    <div className="rounded-2xl border border-slate-100 bg-slate-50 p-5">
      <div className="flex items-center justify-between gap-4">
        <div>
          <p className="text-sm font-bold text-slate-700">
            {label}
          </p>

          <p className="mt-1 text-xs text-slate-400">
            {description}
          </p>
        </div>

        <span className="rounded-lg bg-white px-3 py-1.5 text-xs font-bold text-blue-600 shadow-sm">
          {value}%
        </span>
      </div>

      <input
        type="range"
        min="0"
        max="100"
        value={value}
        onChange={(event) =>
          onChange(Number(event.target.value))
        }
        className="mt-5 w-full accent-blue-600"
      />

      <div className="mt-2 flex justify-between text-[10px] font-medium text-slate-400">
        <span>Low</span>
        <span>Balanced</span>
        <span>High</span>
      </div>
    </div>
  );
}

/* ========================================================= */
/* SECURITY ROW */
/* ========================================================= */

function SecurityRow({
  icon: Icon,
  title,
  description,
  status,
}) {
  return (
    <div className="flex flex-col gap-4 rounded-2xl border border-slate-100 bg-slate-50 p-5 sm:flex-row sm:items-center sm:justify-between">
      <div className="flex items-start gap-4">
        <div className="rounded-xl bg-white p-3 text-emerald-600 shadow-sm">
          <Icon size={19} />
        </div>

        <div>
          <p className="text-sm font-bold text-slate-700">
            {title}
          </p>

          <p className="mt-1 text-xs leading-relaxed text-slate-400">
            {description}
          </p>
        </div>
      </div>

      <span className="flex items-center gap-2 rounded-full border border-emerald-100 bg-emerald-50 px-3 py-1.5 text-xs font-bold text-emerald-600">
        <span className="h-1.5 w-1.5 rounded-full bg-emerald-500" />
        {status}
      </span>
    </div>
  );
}

/* ========================================================= */
/* STATUS BADGE */
/* ========================================================= */

function StatusBadge({
  icon: Icon,
  label,
}) {
  return (
    <span className="flex items-center gap-2 rounded-full border border-white/10 bg-white/5 px-4 py-2 text-xs font-semibold text-slate-300 backdrop-blur">
      <Icon
        size={15}
        className="text-emerald-300"
      />

      {label}
    </span>
  );
}

/* ========================================================= */
/* STATUS ITEM */
/* ========================================================= */

function StatusItem({
  label,
  status,
  last = false,
}) {
  return (
    <div
      className={`flex items-center justify-between py-3 ${
        last
          ? ""
          : "border-b border-slate-100"
      }`}
    >
      <span className="text-xs font-medium text-slate-500">
        {label}
      </span>

      <span className="flex items-center gap-2 text-xs font-bold text-emerald-600">
        <span className="h-1.5 w-1.5 rounded-full bg-emerald-500" />
        {status}
      </span>
    </div>
  );
}

/* ========================================================= */
/* PROFILE METRIC */
/* ========================================================= */

function ProfileMetric({
  label,
  value,
}) {
  return (
    <div>
      <div className="flex items-center justify-between">
        <span className="text-xs font-medium text-slate-400">
          {label}
        </span>

        <span className="text-xs font-bold text-slate-700">
          {value}%
        </span>
      </div>

      <div className="mt-2 h-2 overflow-hidden rounded-full bg-slate-100">
        <div
          className="h-full rounded-full bg-gradient-to-r from-blue-500 to-violet-500 transition-all duration-500"
          style={{
            width: `${value}%`,
          }}
        />
      </div>
    </div>
  );
}

export default Settings;