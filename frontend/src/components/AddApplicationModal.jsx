
import { useState } from "react";

import {
  X,
  Cloud,
  Plus,
  Cpu,
  Sparkles,
  CheckCircle2,
} from "lucide-react";

import API from "../services/api";

function AddApplicationModal({ isOpen, onClose }) {
  const [formData, setFormData] = useState({
    name: "",
    provider: "AWS",
    cost: "",
    utilization: 65,
  });

  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState("");

  if (!isOpen) return null;

  const estimatedCost = formData.cost
    ? (Number(formData.cost) * 0.82).toFixed(2)
    : "0.00";

  const predictedDecision =
    formData.utilization >= 85
      ? "RESIZE"
      : formData.utilization >= 60
      ? "SWITCH"
      : "KEEP";

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (isSubmitting) return;

    const name = formData.name.trim();
    const cost = Number(formData.cost);
    const utilization = Number(formData.utilization);

    if (!name || !formData.cost || !Number.isFinite(cost) || cost <= 0) {
      setError("Please enter a valid application name and monthly cost.");
      return;
    }

    setIsSubmitting(true);
    setError("");

    try {
      await API.post("/applications", {
        name,
        provider: formData.provider,
        cost,
        optimized_cost: Number((cost * 0.82).toFixed(2)),
        utilization,
        decision: predictedDecision,
        status: "Active",
      });

      setFormData({
        name: "",
        provider: "AWS",
        cost: "",
        utilization: 65,
      });

      onClose();
    } catch (err) {
      setError(
        err.response?.data?.detail ||
          "Unable to add the application. Please check your connection and try again."
      );
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="fixed inset-0 z-[100] overflow-y-auto bg-slate-950/70 p-4 backdrop-blur-md">
      <div className="flex min-h-full items-start justify-center py-4 sm:items-center sm:py-6">
        <div className="relative my-auto flex max-h-[calc(100dvh-2rem)] w-full max-w-2xl flex-col overflow-hidden rounded-3xl border border-white/10 bg-white shadow-2xl sm:max-h-[calc(100dvh-3rem)]">
          {/* Background Glow */}

          <div className="pointer-events-none absolute -right-12 -top-12 h-40 w-40 rounded-full bg-blue-500/10 blur-3xl" />

          <div className="pointer-events-none absolute -left-10 bottom-0 h-32 w-32 rounded-full bg-violet-500/10 blur-3xl" />

          {/* Header */}

          <div className="relative flex shrink-0 items-center justify-between border-b border-slate-200 px-5 py-5 sm:px-7 sm:py-6">
            <div className="flex items-center gap-4">
              <div className="rounded-2xl bg-gradient-to-br from-blue-600 to-violet-600 p-3 text-white shadow-lg">
                <Cloud size={26} />
              </div>

              <div>
                <h2 className="text-xl font-bold text-slate-800 sm:text-2xl">
                  Add Cloud Application
                </h2>

                <p className="text-sm text-slate-500">
                  Connect an application for AI-powered optimization.
                </p>
              </div>
            </div>

            <button
              type="button"
              onClick={onClose}
              disabled={isSubmitting}
              aria-label="Close modal"
              className="shrink-0 rounded-xl p-2 text-slate-500 transition hover:bg-slate-100 hover:text-slate-800 disabled:cursor-not-allowed disabled:opacity-50"
            >
              <X size={22} />
            </button>
          </div>

          {/* Scrollable Form */}

          <form
            onSubmit={handleSubmit}
            className="relative min-h-0 flex-1 space-y-7 overflow-y-auto overscroll-contain p-5 sm:p-7"
          >
            {/* Application Name */}

            <div>
              <label
                htmlFor="application-name"
                className="text-sm font-semibold text-slate-700"
              >
                Application Name
              </label>

              <input
                id="application-name"
                required
                value={formData.name}
                disabled={isSubmitting}
                onChange={(e) => {
                  setFormData({
                    ...formData,
                    name: e.target.value,
                  });
                  setError("");
                }}
                placeholder="Example: KhelGyan API"
                className="mt-2 w-full rounded-xl border border-slate-300 px-4 py-3 outline-none transition focus:border-blue-500 focus:ring-4 focus:ring-blue-500/10 disabled:bg-slate-50"
              />
            </div>

            {/* Cloud Provider */}

            <div>
              <label className="text-sm font-semibold text-slate-700">
                Cloud Provider
              </label>

              <div className="mt-3 grid grid-cols-3 gap-3">
                {["AWS", "GCP", "Azure"].map((provider) => (
                  <button
                    key={provider}
                    type="button"
                    disabled={isSubmitting}
                    onClick={() =>
                      setFormData({
                        ...formData,
                        provider,
                      })
                    }
                    className={`rounded-2xl border p-4 transition ${
                      formData.provider === provider
                        ? "border-blue-500 bg-blue-50 text-blue-700 shadow-md"
                        : "border-slate-200 hover:border-blue-300 hover:bg-slate-50"
                    } disabled:cursor-not-allowed disabled:opacity-50`}
                  >
                    <div className="flex flex-col items-center gap-2">
                      <Cpu size={24} />
                      <span className="font-semibold">{provider}</span>
                    </div>
                  </button>
                ))}
              </div>
            </div>

            {/* Monthly Cost */}

            <div>
              <label
                htmlFor="application-cost"
                className="text-sm font-semibold text-slate-700"
              >
                Monthly Cost (USD)
              </label>

              <input
                id="application-cost"
                required
                type="number"
                min="0.01"
                step="0.01"
                value={formData.cost}
                disabled={isSubmitting}
                onChange={(e) => {
                  setFormData({
                    ...formData,
                    cost: e.target.value,
                  });
                  setError("");
                }}
                placeholder="100"
                className="mt-2 w-full rounded-xl border border-slate-300 px-4 py-3 outline-none transition focus:border-blue-500 focus:ring-4 focus:ring-blue-500/10 disabled:bg-slate-50"
              />
            </div>

            {/* Utilization */}

            <div>
              <div className="flex items-center justify-between">
                <label
                  htmlFor="application-utilization"
                  className="text-sm font-semibold text-slate-700"
                >
                  Resource Utilization
                </label>

                <span className="rounded-full bg-blue-50 px-3 py-1 text-sm font-bold text-blue-700">
                  {formData.utilization}%
                </span>
              </div>

              <input
                id="application-utilization"
                type="range"
                min="10"
                max="100"
                value={formData.utilization}
                disabled={isSubmitting}
                onChange={(e) =>
                  setFormData({
                    ...formData,
                    utilization: Number(e.target.value),
                  })
                }
                className="mt-4 w-full accent-blue-600"
              />
            </div>

            {/* AI Preview */}

            <div className="rounded-2xl border border-blue-200 bg-gradient-to-br from-blue-50 to-violet-50 p-5">
              <div className="flex items-center gap-3">
                <Sparkles className="shrink-0 text-violet-600" />

                <div>
                  <h3 className="font-bold text-slate-800">
                    AI Optimization Preview
                  </h3>

                  <p className="text-sm text-slate-500">
                    Estimated results before deployment.
                  </p>
                </div>
              </div>

              <div className="mt-5 grid grid-cols-2 gap-4">
                <div className="rounded-xl bg-white p-4 shadow-sm">
                  <p className="text-xs text-slate-500">
                    Optimized Cost
                  </p>

                  <p className="mt-1 text-xl font-bold text-emerald-600 sm:text-2xl">
                    ${estimatedCost}
                  </p>
                </div>

                <div className="rounded-xl bg-white p-4 shadow-sm">
                  <p className="text-xs text-slate-500">
                    AI Recommendation
                  </p>

                  <p className="mt-1 text-xl font-bold text-blue-600 sm:text-2xl">
                    {predictedDecision}
                  </p>
                </div>
              </div>

              <div className="mt-5 flex items-center gap-2 text-sm text-slate-600">
                <CheckCircle2
                  size={18}
                  className="shrink-0 text-emerald-500"
                />

                <span>
                  CloudMorph AI predicts a potential optimization of
                  approximately 18%.
                </span>
              </div>
            </div>

            {/* API Error */}

            {error && (
              <div
                role="alert"
                className="rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700"
              >
                {error}
              </div>
            )}

            {/* Buttons */}

            <div className="flex shrink-0 gap-4 pb-1">
              <button
                type="button"
                onClick={onClose}
                disabled={isSubmitting}
                className="flex-1 rounded-xl border border-slate-300 py-3 font-semibold text-slate-700 transition hover:bg-slate-100 disabled:cursor-not-allowed disabled:opacity-50"
              >
                Cancel
              </button>

              <button
                type="submit"
                disabled={isSubmitting}
                className="flex flex-1 items-center justify-center gap-2 rounded-xl bg-gradient-to-r from-blue-600 to-violet-600 py-3 font-semibold text-white transition hover:scale-[1.02] disabled:cursor-not-allowed disabled:opacity-70"
              >
                <Plus size={18} />
                {isSubmitting ? "Adding..." : "Add Application"}
              </button>
            </div>
          </form>
        </div>
      </div>
    </div>
  );
}

export default AddApplicationModal;
