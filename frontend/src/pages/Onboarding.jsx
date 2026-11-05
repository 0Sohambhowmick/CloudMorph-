import { useState } from "react";
import { useNavigate } from "react-router-dom";

import {
  Building2,
  Users,
  Cloud,
  Target,
  ArrowRight,
  ArrowLeft,
  CheckCircle2,
} from "lucide-react";

import cloudMorphLogo from "../assets/cloudmorph-logo.png";

function Onboarding() {
  const navigate = useNavigate();

  const [step, setStep] = useState(1);

  const [formData, setFormData] = useState({
    company: "",
    teamSize: "",
    provider: "AWS",
    goal: "Reduce Costs",
  });

  const updateField = (key, value) => {
    setFormData((prev) => ({
      ...prev,
      [key]: value,
    }));
  };

  const finishOnboarding = () => {
    localStorage.setItem(
      "cloudmorph_onboarding",
      JSON.stringify(formData)
    );

    navigate("/app");
  };

  const progress = (step / 3) * 100;

  return (
    <div className="relative flex min-h-screen items-center justify-center overflow-hidden bg-gradient-to-br from-slate-950 via-blue-950 to-violet-950 px-6 py-10">
      {/* Background Glow */}
      <div className="absolute -left-32 top-0 h-96 w-96 rounded-full bg-blue-500/20 blur-3xl"></div>
      <div className="absolute -right-32 bottom-0 h-96 w-96 rounded-full bg-violet-500/20 blur-3xl"></div>

      <div className="relative w-full max-w-3xl overflow-hidden rounded-3xl border border-white/10 bg-white/10 shadow-2xl backdrop-blur-xl">
        {/* Header */}
        <div className="border-b border-white/10 p-8">
          <div className="flex items-center gap-4">
            <div className="flex h-16 w-16 items-center justify-center rounded-2xl bg-gradient-to-br from-blue-600 to-violet-600 p-2 shadow-lg">
              <img
                src={cloudMorphLogo}
                alt="CloudMorph Logo"
                className="h-full w-full object-contain"
              />
            </div>

            <div>
              <h1 className="text-3xl font-bold text-white">
                Welcome to CloudMorph
              </h1>

              <p className="mt-1 text-slate-300">
                Let's personalize your cloud optimization workspace.
              </p>
            </div>
          </div>

          {/* Progress */}
          <div className="mt-8">
            <div className="mb-2 flex justify-between text-sm text-slate-300">
              <span>Step {step} of 3</span>
              <span>{Math.round(progress)}%</span>
            </div>

            <div className="h-2 overflow-hidden rounded-full bg-white/10">
              <div
                className="h-full rounded-full bg-gradient-to-r from-blue-500 to-violet-500 transition-all duration-500"
                style={{ width: `${progress}%` }}
              ></div>
            </div>
          </div>
        </div>

        {/* Content */}
        <div className="p-8">
          {/* STEP 1 */}
          {step === 1 && (
            <div className="space-y-6">
              <div>
                <div className="mb-3 flex items-center gap-3">
                  <Building2 className="text-blue-400" size={24} />

                  <h2 className="text-2xl font-bold text-white">
                    Organization Details
                  </h2>
                </div>

                <p className="text-slate-300">
                  Tell us about your organization.
                </p>
              </div>

              <div>
                <label className="mb-2 block text-sm font-medium text-slate-300">
                  Company Name
                </label>

                <input
                  type="text"
                  value={formData.company}
                  onChange={(e) =>
                    updateField("company", e.target.value)
                  }
                  placeholder="CloudMorph Labs"
                  className="w-full rounded-xl border border-white/10 bg-white/10 px-4 py-3 text-white outline-none transition focus:border-blue-400 focus:ring-2 focus:ring-blue-500/20"
                />
              </div>

              <div>
                <label className="mb-2 block text-sm font-medium text-slate-300">
                  Team Size
                </label>

                <select
                  value={formData.teamSize}
                  onChange={(e) =>
                    updateField("teamSize", e.target.value)
                  }
                  className="w-full rounded-xl border border-white/10 bg-slate-900 px-4 py-3 text-white outline-none transition focus:border-blue-400"
                >
                  <option value="">Select team size</option>
                  <option>1-10</option>
                  <option>11-50</option>
                  <option>51-200</option>
                  <option>201-1000</option>
                  <option>1000+</option>
                </select>
              </div>
            </div>
          )}

          {/* STEP 2 */}
          {step === 2 && (
            <div className="space-y-6">
              <div>
                <div className="mb-3 flex items-center gap-3">
                  <Cloud className="text-blue-400" size={24} />

                  <h2 className="text-2xl font-bold text-white">
                    Cloud Provider
                  </h2>
                </div>

                <p className="text-slate-300">
                  Choose your primary cloud platform.
                </p>
              </div>

              <div className="grid gap-4 md:grid-cols-3">
                {["AWS", "GCP", "Azure"].map((provider) => (
                  <button
                    key={provider}
                    onClick={() =>
                      updateField("provider", provider)
                    }
                    className={`rounded-2xl border p-5 transition-all ${
                      formData.provider === provider
                        ? "border-blue-500 bg-blue-500/20 text-white shadow-lg shadow-blue-500/20"
                        : "border-white/10 bg-white/5 text-slate-300 hover:border-blue-400"
                    }`}
                  >
                    <Cloud className="mx-auto mb-3" size={30} />

                    <p className="font-semibold">{provider}</p>
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* STEP 3 */}
          {step === 3 && (
            <div className="space-y-6">
              <div>
                <div className="mb-3 flex items-center gap-3">
                  <Target className="text-violet-400" size={24} />

                  <h2 className="text-2xl font-bold text-white">
                    Your Primary Goal
                  </h2>
                </div>

                <p className="text-slate-300">
                  What would you like CloudMorph to help you with?
                </p>
              </div>

              <div className="space-y-3">
                {[
                  "Reduce Costs",
                  "Improve Performance",
                  "Optimize Resources",
                  "Monitor Infrastructure",
                ].map((goal) => (
                  <button
                    key={goal}
                    onClick={() => updateField("goal", goal)}
                    className={`flex w-full items-center justify-between rounded-xl border px-5 py-4 transition ${
                      formData.goal === goal
                        ? "border-violet-500 bg-violet-500/20 text-white"
                        : "border-white/10 bg-white/5 text-slate-300 hover:border-violet-400"
                    }`}
                  >
                    <span>{goal}</span>

                    {formData.goal === goal && (
                      <CheckCircle2 size={20} />
                    )}
                  </button>
                ))}
              </div>

              <div className="rounded-2xl border border-blue-500/20 bg-gradient-to-r from-blue-500/10 to-violet-500/10 p-5">
                <div className="flex items-center gap-3">
                  <Users className="text-blue-400" size={22} />

                  <div>
                    <p className="font-semibold text-white">
                      Workspace Ready
                    </p>

                    <p className="text-sm text-slate-300">
                      Your CloudMorph dashboard will be customized
                      based on these preferences.
                    </p>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="flex items-center justify-between border-t border-white/10 p-6">
          <button
            onClick={() =>
              step === 1 ? navigate("/app") : setStep(step - 1)
            }
            className="flex items-center gap-2 rounded-xl border border-white/10 bg-white/5 px-5 py-3 text-slate-300 transition hover:bg-white/10 hover:text-white"
          >
            <ArrowLeft size={18} />

            {step === 1 ? "Skip" : "Back"}
          </button>

          {step < 3 ? (
            <button
              onClick={() => setStep(step + 1)}
              className="flex items-center gap-2 rounded-xl bg-gradient-to-r from-blue-600 to-violet-600 px-6 py-3 font-semibold text-white transition hover:scale-[1.02]"
            >
              Continue

              <ArrowRight size={18} />
            </button>
          ) : (
            <button
              onClick={finishOnboarding}
              className="flex items-center gap-2 rounded-xl bg-gradient-to-r from-blue-600 to-violet-600 px-6 py-3 font-semibold text-white transition hover:scale-[1.02]"
            >
              Launch Workspace

              <ArrowRight size={18} />
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

export default Onboarding;