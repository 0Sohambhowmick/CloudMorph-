import { useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import {
  Cloud,
  Mail,
  Lock,
  Eye,
  EyeOff,
  ArrowRight,
  Sparkles,
  Loader2,
} from "lucide-react";

import API from "../services/api";

function Login() {
  const navigate = useNavigate();

  const [email, setEmail] = useState(
    localStorage.getItem("cloudmorph_remember") === "true"
      ? localStorage.getItem("cloudmorph_user") || ""
      : ""
  );

  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);

  const [rememberMe, setRememberMe] = useState(
    localStorage.getItem("cloudmorph_remember") === "true"
  );

  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");

  /* ========================================================= */
  /* REDIRECT IF ALREADY LOGGED IN */
  /* ========================================================= */

  useEffect(() => {
    const isAuthenticated =
      localStorage.getItem("cloudmorph_auth") === "true";
    const token = localStorage.getItem("cloudmorph_token");

    if (isAuthenticated && token) {
      navigate("/app", { replace: true });
    }
  }, [navigate]);

  /* ========================================================= */
  /* LOGIN */
  /* ========================================================= */

  const handleLogin = async (e) => {
    e.preventDefault();

    if (isLoading) return;

    const normalizedEmail = email.trim().toLowerCase();

    if (!normalizedEmail || !password) {
      setError("Please enter email and password.");
      return;
    }

    setError("");
    setIsLoading(true);

    try {
      const response = await API.post("/auth/login", {
        email: normalizedEmail,
        password,
      });

      const data = response.data;

      if (!data.access_token || !data.user) {
        throw new Error("Invalid response received from the server.");
      }

      /* Save authentication */

      localStorage.setItem("cloudmorph_token", data.access_token);
      localStorage.setItem("cloudmorph_auth", "true");

      /* Remember email only when requested */

      if (rememberMe) {
        localStorage.setItem("cloudmorph_user", normalizedEmail);
        localStorage.setItem("cloudmorph_remember", "true");
      } else {
        localStorage.removeItem("cloudmorph_user");
        localStorage.setItem("cloudmorph_remember", "false");
      }

      /* Save the authenticated user's profile */

      const userData = {
        ...data.user,
        email: data.user.email || normalizedEmail,
        name: data.user.name || normalizedEmail.split("@")[0],
        company: data.user.company || "CloudMorph Labs",
        role: data.user.role || "Cloud Engineer",
        provider: data.user.provider || "AWS",
      };

      localStorage.setItem(
        "cloudmorph_user_data",
        JSON.stringify(userData)
      );

      /* Redirect based on onboarding status */

      const firstTime =
        localStorage.getItem("cloudmorph_onboarding") !== "true";

      navigate(firstTime ? "/app/onboarding" : "/app", {
        replace: true,
      });
    } catch (err) {
      /* Clear authentication data after a failed login */

      localStorage.removeItem("cloudmorph_token");
      localStorage.removeItem("cloudmorph_auth");
      localStorage.removeItem("cloudmorph_user_data");

      const status = err.response?.status;

      if (status === 401) {
        setError("Invalid email or password. Please try again.");
      } else if (status === 400) {
        setError(
          err.response?.data?.detail || "Unable to log in with these details."
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
        setError(err.message || "Login failed. Please try again.");
      }
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="relative flex min-h-screen items-center justify-center overflow-hidden bg-gradient-to-br from-slate-950 via-blue-950 to-violet-950 px-6">
      {/* Background Glow */}

      <div className="absolute -left-32 top-0 h-96 w-96 rounded-full bg-blue-500/20 blur-3xl" />
      <div className="absolute -right-24 bottom-0 h-96 w-96 rounded-full bg-violet-500/20 blur-3xl" />
      <div className="absolute left-1/3 top-1/2 h-72 w-72 rounded-full bg-cyan-500/10 blur-3xl" />

      {/* Login Card */}

      <div className="relative w-full max-w-md rounded-3xl border border-white/10 bg-white/10 p-8 shadow-2xl backdrop-blur-xl">
        {/* Logo */}

        <div className="text-center">
          <div className="mx-auto flex h-16 w-16 items-center justify-center rounded-2xl bg-gradient-to-br from-blue-500 to-violet-600 text-white shadow-lg shadow-blue-900/40">
            <Cloud size={30} />
          </div>

          <h1 className="mt-5 text-3xl font-bold text-white">
            Welcome Back
          </h1>

          <p className="mt-2 text-sm text-slate-300">
            Sign in to your CloudMorph workspace
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

        {/* Form */}

        <form onSubmit={handleLogin} className="mt-8 space-y-5">
          {/* Email */}

          <div>
            <label
              htmlFor="login-email"
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
                id="login-email"
                type="email"
                autoComplete="email"
                required
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="you@example.com"
                className="w-full rounded-xl border border-white/10 bg-white/10 py-3 pl-11 pr-4 text-white placeholder:text-slate-400 outline-none transition focus:border-blue-400 focus:ring-4 focus:ring-blue-500/20"
              />
            </div>
          </div>

          {/* Password */}

          <div>
            <label
              htmlFor="login-password"
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
                id="login-password"
                type={showPassword ? "text" : "password"}
                autoComplete="current-password"
                required
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="Enter password"
                className="w-full rounded-xl border border-white/10 bg-white/10 py-3 pl-11 pr-11 text-white placeholder:text-slate-400 outline-none transition focus:border-blue-400 focus:ring-4 focus:ring-blue-500/20"
              />

              <button
                type="button"
                onClick={() => setShowPassword((previous) => !previous)}
                aria-label={showPassword ? "Hide password" : "Show password"}
                className="absolute right-4 top-1/2 -translate-y-1/2 text-slate-400 transition hover:text-white"
              >
                {showPassword ? (
                  <EyeOff size={18} />
                ) : (
                  <Eye size={18} />
                )}
              </button>
            </div>
          </div>

          {/* Remember + Forgot */}

          <div className="flex items-center justify-between">
            <label className="flex cursor-pointer items-center gap-2 text-sm text-slate-300">
              <input
                type="checkbox"
                checked={rememberMe}
                onChange={(e) => setRememberMe(e.target.checked)}
                className="accent-blue-500"
              />
              Remember me
            </label>

            <Link
              to="/forgot-password"
              className="text-sm font-semibold text-blue-400 transition hover:text-blue-300"
            >
              Forgot Password?
            </Link>
          </div>

          {/* Login Button */}

          <button
            type="submit"
            disabled={isLoading}
            className="flex w-full items-center justify-center gap-2 rounded-xl bg-gradient-to-r from-blue-600 to-violet-600 py-3 font-semibold text-white shadow-lg transition hover:scale-[1.02] hover:shadow-blue-900/30 disabled:cursor-not-allowed disabled:opacity-80"
          >
            {isLoading ? (
              <>
                <Loader2 size={18} className="animate-spin" />
                Signing In...
              </>
            ) : (
              <>
                Login
                <ArrowRight size={18} />
              </>
            )}
          </button>
        </form>

        {/* Footer */}

        <div className="mt-8 border-t border-white/10 pt-6 text-center">
          <p className="text-sm text-slate-300">
            Don't have an account?{" "}
            <Link
              to="/signup"
              className="font-semibold text-blue-400 transition hover:text-blue-300"
            >
              Create Account
            </Link>
          </p>

          <div className="mt-4 flex items-center justify-center gap-2 text-xs text-slate-400">
            <Sparkles size={14} />
            Secure CloudMorph Workspace
          </div>
        </div>
      </div>
    </div>
  );
}

export default Login;