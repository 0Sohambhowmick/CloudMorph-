import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import {
  Cloud,
  Mail,
  Lock,
  User,
  Eye,
  EyeOff,
  ArrowRight,
  Sparkles,
  CheckCircle2,
  Loader2,
} from "lucide-react";

import API from "../services/api";

function Signup() {
  const navigate = useNavigate();

  const [formData, setFormData] = useState({
    fullName: "",
    email: "",
    password: "",
    confirmPassword: "",
  });

  const [showPassword, setShowPassword] = useState(false);
  const [showConfirm, setShowConfirm] = useState(false);
  const [acceptedTerms, setAcceptedTerms] = useState(false);

  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");

  /* ========================================================= */
  /* HANDLE FORM INPUTS */
  /* ========================================================= */

  const handleChange = (e) => {
    const { name, value } = e.target;

    setFormData((previous) => ({
      ...previous,
      [name]: value,
    }));

    if (error) {
      setError("");
    }
  };

  /* ========================================================= */
  /* SIGNUP */
  /* ========================================================= */

  const handleSignup = async (e) => {
    e.preventDefault();

    if (isLoading) return;

    setError("");

    const name = formData.fullName.trim();
    const email = formData.email.trim().toLowerCase();
    const password = formData.password;
    const confirmPassword = formData.confirmPassword;

    if (!name || !email || !password || !confirmPassword) {
      setError("Please fill in all required fields.");
      return;
    }

    if (password !== confirmPassword) {
      setError("Passwords do not match.");
      return;
    }

    if (!acceptedTerms) {
      setError("Please accept the Terms & Conditions.");
      return;
    }

    setIsLoading(true);

    try {
      /* Send registration details to FastAPI */

      const response = await API.post("/auth/signup", {
        name,
        email,
        password,
      });

      const data = response.data;

      if (!data.access_token || !data.user) {
        throw new Error("Invalid response received from the server.");
      }

      /* Save authentication token */

      localStorage.setItem("cloudmorph_token", data.access_token);
      localStorage.setItem("cloudmorph_auth", "true");

      /* Save the registered email */

      localStorage.setItem("cloudmorph_user", email);
      localStorage.setItem("cloudmorph_remember", "true");

      /* Save the user profile returned by the backend */

      const userData = {
        ...data.user,
        name: data.user.name || name,
        email: data.user.email || email,
        company: data.user.company || "CloudMorph Labs",
        role: data.user.role || "Cloud Engineer",
        provider: data.user.provider || "AWS",
      };

      localStorage.setItem(
        "cloudmorph_user_data",
        JSON.stringify(userData)
      );

      /* Start onboarding for the newly registered user */

      localStorage.removeItem("cloudmorph_onboarding");

      navigate("/app/onboarding", { replace: true });
    } catch (err) {
      /* Clear potentially stale authentication data */

      localStorage.removeItem("cloudmorph_token");
      localStorage.removeItem("cloudmorph_auth");
      localStorage.removeItem("cloudmorph_user_data");

      const status = err.response?.status;

      if (status === 400) {
        setError(
          err.response?.data?.detail ||
            "This email may already be registered. Please try logging in."
        );
      } else if (status === 422) {
        const detail = err.response?.data?.detail;

        setError(
          Array.isArray(detail)
            ? detail.map((item) => item.msg).join(", ")
            : typeof detail === "string"
              ? detail
              : "Please check your registration details."
        );
      } else if (err.response?.data?.detail) {
        setError(err.response.data.detail);
      } else if (err.response) {
        setError("The server encountered an error. Please try again.");
      } else if (err.request) {
        setError(
          "Unable to connect to CloudMorph API. Check that the backend is running."
        );
      } else {
        setError(err.message || "Signup failed. Please try again.");
      }
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="relative flex min-h-screen items-center justify-center overflow-hidden bg-gradient-to-br from-slate-950 via-blue-950 to-violet-950 px-6 py-8">
      {/* Background Effects */}

      <div className="absolute -left-32 top-0 h-96 w-96 rounded-full bg-blue-500/20 blur-3xl" />
      <div className="absolute -right-24 bottom-0 h-96 w-96 rounded-full bg-violet-500/20 blur-3xl" />
      <div className="absolute left-1/3 top-1/2 h-72 w-72 rounded-full bg-cyan-500/10 blur-3xl" />

      {/* Signup Card */}

      <div className="relative w-full max-w-md rounded-3xl border border-white/10 bg-white/10 p-8 shadow-2xl backdrop-blur-xl">
        {/* Logo */}

        <div className="text-center">
          <div className="mx-auto flex h-16 w-16 items-center justify-center rounded-2xl bg-gradient-to-br from-blue-500 to-violet-600 text-white shadow-lg">
            <Cloud size={30} />
          </div>

          <h1 className="mt-5 text-3xl font-bold text-white">
            Join CloudMorph
          </h1>

          <p className="mt-2 text-sm text-slate-300">
            Create your AI-powered cloud optimization workspace.
          </p>
        </div>

        {/* Error Message */}

        {error && (
          <div
            role="alert"
            aria-live="polite"
            className="mt-6 rounded-xl border border-red-400/20 bg-red-500/10 px-4 py-3 text-sm text-red-200 backdrop-blur"
          >
            {error}
          </div>
        )}

        {/* Signup Form */}

        <form onSubmit={handleSignup} className="mt-8 space-y-5">
          {/* Full Name */}

          <div>
            <label
              htmlFor="signup-fullName"
              className="text-sm font-semibold text-slate-200"
            >
              Full Name
            </label>

            <div className="relative mt-2">
              <User
                size={18}
                className="absolute left-4 top-1/2 -translate-y-1/2 text-slate-400"
              />

              <input
                id="signup-fullName"
                type="text"
                name="fullName"
                autoComplete="name"
                required
                value={formData.fullName}
                onChange={handleChange}
                placeholder="Soham Bhowmick"
                className="w-full rounded-xl border border-white/10 bg-white/10 py-3 pl-11 pr-4 text-white placeholder:text-slate-400 outline-none transition focus:border-blue-400 focus:ring-4 focus:ring-blue-500/20"
              />
            </div>
          </div>

          {/* Email */}

          <div>
            <label
              htmlFor="signup-email"
              className="text-sm font-semibold text-slate-200"
            >
              Email
            </label>

            <div className="relative mt-2">
              <Mail
                size={18}
                className="absolute left-4 top-1/2 -translate-y-1/2 text-slate-400"
              />

              <input
                id="signup-email"
                type="email"
                name="email"
                autoComplete="email"
                required
                value={formData.email}
                onChange={handleChange}
                placeholder="you@example.com"
                className="w-full rounded-xl border border-white/10 bg-white/10 py-3 pl-11 pr-4 text-white placeholder:text-slate-400 outline-none transition focus:border-blue-400 focus:ring-4 focus:ring-blue-500/20"
              />
            </div>
          </div>

          {/* Password */}

          <div>
            <label
              htmlFor="signup-password"
              className="text-sm font-semibold text-slate-200"
            >
              Password
            </label>

            <div className="relative mt-2">
              <Lock
                size={18}
                className="absolute left-4 top-1/2 -translate-y-1/2 text-slate-400"
              />

              <input
                id="signup-password"
                type={showPassword ? "text" : "password"}
                name="password"
                autoComplete="new-password"
                required
                value={formData.password}
                onChange={handleChange}
                placeholder="Create password"
                className="w-full rounded-xl border border-white/10 bg-white/10 py-3 pl-11 pr-11 text-white placeholder:text-slate-400 outline-none transition focus:border-blue-400 focus:ring-4 focus:ring-blue-500/20"
              />

              <button
                type="button"
                onClick={() => setShowPassword((previous) => !previous)}
                aria-label={showPassword ? "Hide password" : "Show password"}
                className="absolute right-4 top-1/2 -translate-y-1/2 text-slate-400 hover:text-white"
              >
                {showPassword ? (
                  <EyeOff size={18} />
                ) : (
                  <Eye size={18} />
                )}
              </button>
            </div>
          </div>

          {/* Confirm Password */}

          <div>
            <label
              htmlFor="signup-confirmPassword"
              className="text-sm font-semibold text-slate-200"
            >
              Confirm Password
            </label>

            <div className="relative mt-2">
              <Lock
                size={18}
                className="absolute left-4 top-1/2 -translate-y-1/2 text-slate-400"
              />

              <input
                id="signup-confirmPassword"
                type={showConfirm ? "text" : "password"}
                name="confirmPassword"
                autoComplete="new-password"
                required
                value={formData.confirmPassword}
                onChange={handleChange}
                placeholder="Confirm password"
                className="w-full rounded-xl border border-white/10 bg-white/10 py-3 pl-11 pr-11 text-white placeholder:text-slate-400 outline-none transition focus:border-blue-400 focus:ring-4 focus:ring-blue-500/20"
              />

              <button
                type="button"
                onClick={() => setShowConfirm((previous) => !previous)}
                aria-label={
                  showConfirm ? "Hide confirmation password" : "Show confirmation password"
                }
                className="absolute right-4 top-1/2 -translate-y-1/2 text-slate-400 hover:text-white"
              >
                {showConfirm ? (
                  <EyeOff size={18} />
                ) : (
                  <Eye size={18} />
                )}
              </button>
            </div>
          </div>

          {/* Terms */}

          <label className="flex items-start gap-3 text-sm text-slate-300">
            <input
              type="checkbox"
              checked={acceptedTerms}
              onChange={(e) => {
                setAcceptedTerms(e.target.checked);
                if (error) setError("");
              }}
              className="mt-1 accent-blue-500"
            />

            <span>
              I agree to the{" "}
              <span className="font-semibold text-blue-400">
                Terms &amp; Conditions
              </span>{" "}
              and{" "}
              <span className="font-semibold text-blue-400">
                Privacy Policy
              </span>
              .
            </span>
          </label>

          {/* Signup Button */}

          <button
            type="submit"
            disabled={isLoading}
            className="flex w-full items-center justify-center gap-2 rounded-xl bg-gradient-to-r from-blue-600 to-violet-600 py-3 font-semibold text-white transition hover:scale-[1.02] disabled:cursor-not-allowed disabled:opacity-80"
          >
            {isLoading ? (
              <>
                <Loader2 size={18} className="animate-spin" />
                Creating Workspace...
              </>
            ) : (
              <>
                Create Workspace
                <ArrowRight size={18} />
              </>
            )}
          </button>
        </form>

        {/* Footer */}

        <div className="mt-8 border-t border-white/10 pt-6 text-center">
          <p className="text-sm text-slate-300">
            Already have an account?{" "}
            <Link
              to="/login"
              className="font-semibold text-blue-400 hover:text-blue-300"
            >
              Sign In
            </Link>
          </p>

          <div className="mt-4 flex items-center justify-center gap-2 text-xs text-slate-400">
            <CheckCircle2 size={14} />
            Enterprise-ready Cloud Management
          </div>
        </div>
      </div>
    </div>
  );
}

export default Signup;