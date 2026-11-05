import { useState } from "react";
import { useNavigate } from "react-router-dom";
import {
  Cloud,
  CheckCircle2,
  ArrowLeft,
  ArrowRight,
  Sparkles,
  Loader2,
} from "lucide-react";

import API from "../services/api";
import cloudMorphLogo from "../assets/cloudmorph-logo.png";

function CloudProviderSetup() {
  const navigate = useNavigate();

  const [selectedProvider, setSelectedProvider] = useState("AWS");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");

  const providers = [
    {
      name: "AWS",
      color: "from-orange-500 to-amber-500",
      bg: "border-orange-500/30 bg-orange-500/10",
      description: "Amazon Web Services",
    },
    {
      name: "Google Cloud",
      color: "from-blue-500 to-cyan-500",
      bg: "border-blue-500/30 bg-blue-500/10",
      description: "Google Cloud Platform",
    },
    {
      name: "Azure",
      color: "from-sky-500 to-indigo-500",
      bg: "border-sky-500/30 bg-sky-500/10",
      description: "Microsoft Azure",
    },
  ];

  const handleContinue = async () => {
    setError("");
    setIsLoading(true);

    try {
      // Save provider to backend
      const response = await API.put("/auth/profile", {
        provider: selectedProvider,
      });

      // Update local user profile
      if (response.data.user) {
        localStorage.setItem(
          "cloudmorph_user_data",
          JSON.stringify(response.data.user)
        );
      }

      // Preserve onboarding data
      const existing = JSON.parse(
        localStorage.getItem("cloudmorph_onboarding") || "{}"
      );

      localStorage.setItem(
        "cloudmorph_onboarding",
        JSON.stringify({
          ...existing,
          cloudProvider: selectedProvider,
          providerSetup: true,
        })
      );

      // Continue onboarding flow
      navigate("/app/onboarding/ai-preferences");
    } catch (err) {
      setError(
        err.response?.data?.detail ||
          "Unable to save your cloud provider. Please try again."
      );
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="relative min-h-screen overflow-hidden bg-gradient-to-br from-slate-950 via-blue-950 to-violet-950 text-white">
      {/* Background Effects */}
      <div className="absolute -left-32 top-0 h-[420px] w-[420px] rounded-full bg-blue-600/20 blur-[120px]" />
      <div className="absolute -right-24 bottom-0 h-[420px] w-[420px] rounded-full bg-violet-600/20 blur-[120px]" />

      {/* Header */}
      <header className="border-b border-white/10 bg-slate-950/40 backdrop-blur-xl">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-8 py-6">
          <div className="flex items-center gap-4">
            <div className="flex h-16 w-16 items-center justify-center rounded-2xl border border-blue-500/20 bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 shadow-[0_0_30px_rgba(37,99,235,0.35)]">
              <img
                src={cloudMorphLogo}
                alt="CloudMorph"
                className="h-11 w-11 object-contain drop-shadow-[0_0_12px_rgba(59,130,246,0.6)]"
              />
            </div>

            <div>
              <h1 className="text-3xl font-bold">CloudMorph</h1>
              <p className="text-sm text-slate-400">
                Adaptive Cloud Intelligence
              </p>
            </div>
          </div>

          <span className="rounded-full border border-emerald-500/30 bg-emerald-500/10 px-4 py-2 text-sm font-semibold text-emerald-300">
            Cloud Provider Setup
          </span>
        </div>
      </header>

      {/* Main */}
      <main className="relative mx-auto max-w-6xl px-6 py-12">
        <div className="text-center">
          <div className="inline-flex items-center gap-2 rounded-full border border-blue-500/30 bg-blue-500/10 px-5 py-2 text-sm font-semibold text-blue-300">
            <Sparkles size={16} />
            Step 2 of 4
          </div>

          <h2 className="mt-6 text-5xl font-black">
            Choose Your Cloud Provider
          </h2>

          <p className="mx-auto mt-4 max-w-2xl text-lg text-slate-300">
            Select the cloud platform CloudMorph will optimize first.
            Additional providers can be connected later.
          </p>
        </div>

        {/* Error */}
        {error && (
          <div className="mx-auto mt-8 max-w-2xl rounded-2xl border border-red-500/30 bg-red-500/10 px-6 py-4 text-center text-sm text-red-200 backdrop-blur">
            {error}
          </div>
        )}

        {/* Provider Cards */}
        <div className="mt-14 grid gap-8 md:grid-cols-3">
          {providers.map((provider) => (
            <button
              key={provider.name}
              onClick={() => !isLoading && setSelectedProvider(provider.name)}
              disabled={isLoading}
              className={`group relative overflow-hidden rounded-3xl border p-8 text-left transition-all duration-300 hover:-translate-y-2 hover:shadow-2xl ${
                selectedProvider === provider.name
                  ? provider.bg + " scale-[1.02]"
                  : "border-white/10 bg-white/5 hover:border-blue-500/30"
              } ${isLoading ? "cursor-not-allowed opacity-70" : ""}`}
            >
              {selectedProvider === provider.name && (
                <div className="absolute right-5 top-5">
                  <CheckCircle2 className="text-white" size={24} />
                </div>
              )}

              <div
                className={`inline-flex rounded-2xl bg-gradient-to-r ${provider.color} p-4 shadow-lg`}
              >
                <Cloud size={32} />
              </div>

              <h3 className="mt-6 text-2xl font-bold">{provider.name}</h3>

              <p className="mt-2 text-sm text-slate-300">
                {provider.description}
              </p>

              <div className="mt-8 rounded-xl bg-slate-900/60 p-4 text-sm text-slate-300">
                CloudMorph will analyze utilization, cost, and AI optimization
                opportunities for {provider.name}.
              </div>
            </button>
          ))}
        </div>

        {/* Footer Buttons */}
        <div className="mt-12 flex items-center justify-between border-t border-white/10 pt-8">
          <button
            onClick={() => navigate("/app/onboarding")}
            disabled={isLoading}
            className="flex items-center gap-2 rounded-2xl border border-white/10 bg-white/5 px-6 py-3 font-semibold transition hover:bg-white/10 disabled:cursor-not-allowed disabled:opacity-60"
          >
            <ArrowLeft size={18} />
            Back
          </button>

          <button
            onClick={handleContinue}
            disabled={isLoading}
            className="flex items-center gap-2 rounded-2xl bg-gradient-to-r from-blue-600 to-violet-600 px-8 py-4 font-bold shadow-xl shadow-blue-900/30 transition hover:scale-[1.03] disabled:cursor-not-allowed disabled:opacity-80"
          >
            {isLoading ? (
              <>
                <Loader2 size={18} className="animate-spin" />
                Saving...
              </>
            ) : (
              <>
                Continue
                <ArrowRight size={18} />
              </>
            )}
          </button>
        </div>
      </main>
    </div>
  );
}

export default CloudProviderSetup;