import { useCallback, useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import {
  Camera,
  Edit3,
  ShieldCheck,
  User,
  Building2,
  CalendarDays,
  Save,
  Lock,
  CheckCircle2,
  Cloud,
  Sparkles,
  Award,
  Activity,
  Globe,
  RefreshCw,
} from "lucide-react";
import API from "../services/api";

function readStoredUser() {
  try {
    const parsed = JSON.parse(localStorage.getItem("cloudmorph_user_data") || "{}");
    return parsed && typeof parsed === "object" ? parsed : {};
  } catch {
    return {};
  }
}

function getStoredEmail(userData) {
  return (
    userData.email ||
    localStorage.getItem("cloudmorph_email") ||
    localStorage.getItem("cloudmorph_user") ||
    "user@cloudmorph.ai"
  );
}

function toNumber(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number : 0;
}

function getAppId(app) {
  return String(app?.id ?? app?._id ?? "");
}

function Profile() {
  const navigate = useNavigate();
  const storedUser = useMemo(() => readStoredUser(), []);
  const email = useMemo(() => getStoredEmail(storedUser), [storedUser]);

  const [editing, setEditing] = useState(false);
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const [error, setError] = useState("");
  const [name, setName] = useState(() =>
    localStorage.getItem("cloudmorph_name") ||
      storedUser.name ||
      storedUser.full_name ||
      storedUser.fullName ||
      storedUser.username ||
      (email.includes("@") ? email.split("@")[0].replace(/[._-]+/g, " ") : "") ||
      "CloudMorph User"
  );
  const [workspace, setWorkspace] = useState(
    () => localStorage.getItem("cloudmorph_workspace") || "CloudMorph Workspace"
  );
  const [company, setCompany] = useState(
    () => localStorage.getItem("cloudmorph_company") || "Personal Workspace"
  );
  const [website, setWebsite] = useState(
    () => localStorage.getItem("cloudmorph_website") || "cloudmorph.ai"
  );
  const [statistics, setStatistics] = useState({
    resources: null,
    recommendations: null,
    savings: null,
  });
  const [statsLoading, setStatsLoading] = useState(true);

  const role = "Workspace Administrator";

  const initials = useMemo(() => {
    const words = String(name || "CloudMorph User").trim().split(/\s+/).filter(Boolean);
    return words.map((word) => word[0]).join("").slice(0, 2).toUpperCase() || "CM";
  }, [name]);

  const fetchStatistics = useCallback(async () => {
    setStatsLoading(true);
    try {
      const response = await API.get("/applications");
      const payload = response.data;
      const apps = Array.isArray(payload)
        ? payload
        : Array.isArray(payload?.applications)
          ? payload.applications
          : Array.isArray(payload?.data)
            ? payload.data
            : [];

      const normalized = apps.filter((app) => app && typeof app === "object");
      const recommendations = normalized.filter((app) => {
        const decision = String(app.decision || "").trim().toUpperCase();
        return decision && !["ANALYZE", "NONE", "PENDING"].includes(decision);
      }).length;
      const savings = normalized.reduce((total, app) => {
        const cost = Math.max(0, toNumber(app.cost ?? app.monthly_cost ?? app.monthlyCost));
        const optimized = toNumber(
          app.optimized_cost ?? app.optimizedCost ?? app.optimized_cost_monthly
        );
        return total + (optimized > 0 ? Math.max(0, cost - Math.min(optimized, cost)) : 0);
      }, 0);

      setStatistics({
        resources: normalized.length,
        recommendations,
        savings,
      });
    } catch (fetchError) {
      console.error("Unable to load profile statistics:", fetchError);
      setStatistics({ resources: null, recommendations: null, savings: null });
    } finally {
      setStatsLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchStatistics();
  }, [fetchStatistics]);

  const handleSave = () => {
    const trimmedName = name.trim();
    const trimmedWorkspace = workspace.trim();
    const trimmedCompany = company.trim();
    const trimmedWebsite = website.trim();

    if (!trimmedName) {
      setError("Full name cannot be empty.");
      return;
    }

    setSaving(true);
    setSaved(false);
    setError("");

    try {
      localStorage.setItem("cloudmorph_name", trimmedName);
      localStorage.setItem("cloudmorph_workspace", trimmedWorkspace);
      localStorage.setItem("cloudmorph_company", trimmedCompany);
      localStorage.setItem("cloudmorph_website", trimmedWebsite);

      // Keep the locally stored user profile in sync when available.
      const currentUser = readStoredUser();
      const updatedUser = {
        ...currentUser,
        name: trimmedName,
        full_name: trimmedName,
        email,
      };
      localStorage.setItem("cloudmorph_user_data", JSON.stringify(updatedUser));

      setName(trimmedName);
      setWorkspace(trimmedWorkspace);
      setCompany(trimmedCompany);
      setWebsite(trimmedWebsite);
      setEditing(false);
      setSaved(true);
      window.setTimeout(() => setSaved(false), 2500);
    } catch (saveError) {
      console.error("Unable to save profile:", saveError);
      setError("Unable to save your profile in this browser. Please try again.");
    } finally {
      setSaving(false);
    }
  };

  const cancelEditing = () => {
    setName(
      localStorage.getItem("cloudmorph_name") ||
        storedUser.name ||
        storedUser.full_name ||
        storedUser.fullName ||
        storedUser.username ||
        (email.includes("@") ? email.split("@")[0].replace(/[._-]+/g, " ") : "") ||
        "CloudMorph User"
    );
    setWorkspace(localStorage.getItem("cloudmorph_workspace") || "CloudMorph Workspace");
    setCompany(localStorage.getItem("cloudmorph_company") || "Personal Workspace");
    setWebsite(localStorage.getItem("cloudmorph_website") || "cloudmorph.ai");
    setError("");
    setEditing(false);
  };

  return (
    <div className="min-h-screen bg-slate-100">
      {/* HERO */}
      <section className="relative overflow-hidden bg-gradient-to-br from-slate-950 via-blue-950 to-violet-950">
        <div className="absolute -right-20 -top-20 h-96 w-96 rounded-full bg-blue-500/20 blur-3xl" />
        <div className="absolute bottom-0 left-1/3 h-72 w-72 rounded-full bg-violet-500/20 blur-3xl" />

        <div className="relative px-8 py-10">
          <div className="flex flex-col gap-6 lg:flex-row lg:items-center lg:justify-between">
            <div className="flex items-center gap-5">
              <div className="relative">
                <div className="flex h-28 w-28 items-center justify-center rounded-full bg-gradient-to-br from-blue-500 to-violet-600 text-3xl font-bold text-white shadow-xl">
                  {initials}
                </div>
                <button
                  type="button"
                  disabled
                  aria-label="Profile photo upload coming soon"
                  title="Profile photo upload coming soon"
                  className="absolute bottom-1 right-1 cursor-not-allowed rounded-full bg-white p-2 text-slate-700 shadow-lg opacity-70"
                >
                  <Camera size={18} />
                </button>
              </div>

              <div>
                <p className="text-xs font-bold uppercase tracking-wider text-blue-200">
                  CloudMorph Account
                </p>
                <h1 className="mt-2 text-3xl font-bold text-white md:text-5xl">{name}</h1>
                <p className="mt-2 text-slate-300">{email}</p>
                <div className="mt-4 flex flex-wrap gap-3">
                  <Badge icon={ShieldCheck} text={role} />
                  <Badge icon={Cloud} text="Workspace Active" />
                </div>
              </div>
            </div>

            <div className="flex flex-wrap gap-3">
              {editing && (
                <button
                  type="button"
                  onClick={cancelEditing}
                  disabled={saving}
                  className="flex items-center gap-2 rounded-xl border border-white/20 bg-white/10 px-5 py-3 font-bold text-white transition hover:bg-white/20 disabled:opacity-70"
                >
                  Cancel
                </button>
              )}
              <button
                type="button"
                onClick={() => (editing ? handleSave() : setEditing(true))}
                disabled={saving}
                className="flex items-center gap-2 rounded-xl bg-white px-5 py-3 font-bold text-slate-900 transition hover:bg-slate-100 disabled:opacity-70"
              >
                {saving ? (
                  <RefreshCw size={18} className="animate-spin" />
                ) : saved ? (
                  <CheckCircle2 size={18} />
                ) : editing ? (
                  <Save size={18} />
                ) : (
                  <Edit3 size={18} />
                )}
                {saving ? "Saving..." : saved ? "Saved" : editing ? "Save Changes" : "Edit Profile"}
              </button>
            </div>
          </div>
        </div>
      </section>

      {/* MAIN */}
      <main className="p-8">
        {error && (
          <div role="alert" className="mb-6 rounded-2xl border border-red-200 bg-red-50 p-4 text-sm text-red-700">
            {error}
          </div>
        )}
        <div className="grid grid-cols-1 gap-8 xl:grid-cols-3">
          {/* LEFT */}
          <div className="space-y-8 xl:col-span-2">
            <Section title="Personal Information" icon={User}>
              <Field label="Full Name" value={name} editable={editing} onChange={setName} />
              <Field label="Email Address" value={email} />
              <Field label="Role" value={role} />
            </Section>

            <Section title="Workspace" icon={Building2}>
              <Field label="Workspace Name" value={workspace} editable={editing} onChange={setWorkspace} />
              <Field label="Organization" value={company} editable={editing} onChange={setCompany} />
              <Field label="Website" value={website} editable={editing} onChange={setWebsite} icon={Globe} />
            </Section>

            <Section title="Security" icon={Lock}>
              <SecurityItem
                title="Password"
                description="Manage your CloudMorph password."
                button="Change Password"
                onClick={() => navigate("/forgot-password")}
              />
              <SecurityItem
                title="Two-Factor Authentication"
                description="Will be available after backend integration."
                button="Coming Soon"
                disabled
              />
              <SecurityItem
                title="Active Sessions"
                description="Current device only."
                button="View Sessions"
                disabled
              />
            </Section>
          </div>

          {/* RIGHT */}
          <aside className="space-y-8">
            <div className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm">
              <div className="flex items-center gap-3">
                <div className="rounded-xl bg-emerald-50 p-3 text-emerald-600">
                  <CheckCircle2 size={22} />
                </div>
                <div>
                  <h3 className="font-bold text-slate-800">Account Status</h3>
                  <p className="text-xs text-slate-400">Status of your current workspace</p>
                </div>
              </div>
              <div className="mt-6 space-y-4">
                <Status label="Workspace" value="Active" />
                <Status label="Authentication" value="Enabled" />
                <Status label="AI Engine" value="Decision service" />
              </div>
            </div>

            <div className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm">
              <div className="flex items-center justify-between gap-3">
                <h3 className="font-bold text-slate-800">Account Statistics</h3>
                <button
                  type="button"
                  onClick={fetchStatistics}
                  disabled={statsLoading}
                  aria-label="Refresh account statistics"
                  title="Refresh statistics"
                  className="rounded-lg p-2 text-slate-500 transition hover:bg-slate-100 hover:text-blue-600 disabled:opacity-50"
                >
                  <RefreshCw size={16} className={statsLoading ? "animate-spin" : ""} />
                </button>
              </div>
              <div className="mt-6 space-y-4">
                <Stat icon={Activity} label="Applications" value={statsLoading ? "Loading..." : statistics.resources ?? "Unavailable"} />
                <Stat icon={Sparkles} label="Saved Decisions" value={statsLoading ? "Loading..." : statistics.recommendations ?? "Unavailable"} />
                <Stat
                  icon={Award}
                  label="Estimated Savings"
                  value={statsLoading ? "Loading..." : statistics.savings === null ? "Unavailable" : `$${statistics.savings.toFixed(2)}`}
                />
              </div>
              {!statsLoading && statistics.resources === null && (
                <p className="mt-4 text-xs text-slate-400">Statistics could not be loaded. Check your connection and try refreshing.</p>
              )}
            </div>

            <div className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm">
              <div className="flex items-center gap-3">
                <div className="rounded-xl bg-blue-50 p-3 text-blue-600">
                  <CalendarDays size={22} />
                </div>
                <div>
                  <h3 className="font-bold text-slate-800">Member Since</h3>
                  <p className="text-sm text-slate-500">{storedUser.created_at ? new Date(storedUser.created_at).toLocaleDateString(undefined, { month: "long", year: "numeric" }) : "Not available"}</p>
                </div>
              </div>
            </div>
          </aside>
        </div>
      </main>
    </div>
  );
}

/* ---------- Components ---------- */

function Badge({ icon: Icon, text }) {
  return (
    <span className="flex items-center gap-2 rounded-full border border-white/10 bg-white/5 px-4 py-2 text-xs font-semibold text-slate-300 backdrop-blur">
      <Icon size={14} className="text-blue-300" />
      {text}
    </span>
  );
}

function Section({ title, icon: Icon, children }) {
  return (
    <section className="rounded-3xl border border-slate-200 bg-white p-7 shadow-sm">
      <div className="flex items-center gap-3">
        <div className="rounded-xl bg-blue-50 p-3 text-blue-600">
          <Icon size={21} />
        </div>
        <h2 className="text-xl font-bold text-slate-800">{title}</h2>
      </div>
      <div className="mt-7 space-y-5">{children}</div>
    </section>
  );
}

function Field({ label, value, editable = false, onChange, icon: Icon }) {
  return (
    <div>
      <label className="text-xs font-bold text-slate-500">{label}</label>
      {editable && onChange ? (
        <div className="relative mt-2">
          {Icon && <Icon size={17} className="absolute left-4 top-1/2 -translate-y-1/2 text-slate-400" />}
          <input
            value={value}
            onChange={(event) => onChange(event.target.value)}
            aria-label={label}
            className={`w-full rounded-xl border border-slate-200 bg-slate-50 px-4 py-3 outline-none focus:border-blue-400 focus:ring-4 focus:ring-blue-100 ${Icon ? "pl-11" : ""}`}
          />
        </div>
      ) : (
        <div className="mt-2 rounded-xl bg-slate-50 px-4 py-3 font-medium text-slate-700">{value}</div>
      )}
    </div>
  );
}

function SecurityItem({ title, description, button, onClick, disabled = false }) {
  return (
    <div className="flex flex-col gap-3 rounded-2xl border border-slate-100 bg-slate-50 p-5 sm:flex-row sm:items-center sm:justify-between">
      <div>
        <p className="font-bold text-slate-700">{title}</p>
        <p className="text-sm text-slate-500">{description}</p>
      </div>
      <button
        type="button"
        onClick={onClick}
        disabled={disabled}
        className="rounded-xl bg-slate-900 px-4 py-2 text-sm font-bold text-white transition hover:bg-blue-600 disabled:cursor-not-allowed disabled:bg-slate-300"
      >
        {button}
      </button>
    </div>
  );
}

function Status({ label, value }) {
  return (
    <div className="flex items-center justify-between gap-3">
      <span className="text-sm text-slate-500">{label}</span>
      <span className="flex items-center gap-2 text-sm font-bold text-emerald-600">
        <span className="h-2 w-2 rounded-full bg-emerald-500" />
        {value}
      </span>
    </div>
  );
}

function Stat({ icon: Icon, label, value }) {
  return (
    <div className="flex items-center justify-between gap-3 rounded-xl bg-slate-50 p-4">
      <div className="flex items-center gap-3">
        <div className="rounded-lg bg-white p-2 text-blue-600 shadow-sm">
          <Icon size={18} />
        </div>
        <span className="text-sm text-slate-600">{label}</span>
      </div>
      <span className="text-right font-bold text-slate-800">{value}</span>
    </div>
  );
}

export default Profile;
