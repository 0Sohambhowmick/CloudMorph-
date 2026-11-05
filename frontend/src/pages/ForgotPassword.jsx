import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import {
  ArrowLeft,
  Mail,
  Send,
  Cloud,
  Sparkles,
  CheckCircle2,
} from "lucide-react";

function ForgotPassword() {
  const navigate = useNavigate();

  const [email, setEmail] = useState("");
  const [sent, setSent] = useState(false);

  const handleSubmit = (e) => {
    e.preventDefault();

    if (!email.trim()) return;

    // Temporary demo behavior
    setSent(true);

    setTimeout(() => {
      navigate("/login");
    }, 2500);
  };

  return (
    <div className="relative flex min-h-screen items-center justify-center overflow-hidden bg-gradient-to-br from-slate-950 via-blue-950 to-violet-950 px-6">
      {/* Background Glow */}
      <div className="absolute -left-24 top-0 h-96 w-96 rounded-full bg-blue-500/20 blur-3xl"></div>
      <div className="absolute -right-20 bottom-0 h-96 w-96 rounded-full bg-violet-500/20 blur-3xl"></div>

      <div className="relative w-full max-w-md rounded-3xl border border-white/10 bg-white/10 p-8 shadow-2xl backdrop-blur-xl">
        {/* Logo */}
        <div className="mb-8 text-center">
          <div className="mx-auto flex h-16 w-16 items-center justify-center rounded-2xl bg-gradient-to-br from-blue-500 to-violet-600 text-white shadow-lg">
            <Cloud size={30} />
          </div>

          <h1 className="mt-5 text-3xl font-bold text-white">
            CloudMorph
          </h1>

          <p className="mt-2 text-sm text-slate-300">
            Reset your workspace password
          </p>
        </div>

        {!sent ? (
          <form onSubmit={handleSubmit} className="space-y-6">
            <div>
              <label className="text-sm font-semibold text-slate-200">
                Email Address
              </label>

              <div className="relative mt-2">
                <Mail
                  size={18}
                  className="absolute left-4 top-1/2 -translate-y-1/2 text-slate-400"
                />

                <input
                  type="email"
                  required
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  placeholder="you@example.com"
                  className="w-full rounded-xl border border-white/10 bg-white/10 py-3 pl-11 pr-4 text-white placeholder:text-slate-400 outline-none transition focus:border-blue-400 focus:ring-4 focus:ring-blue-500/20"
                />
              </div>
            </div>

            <button
              type="submit"
              className="flex w-full items-center justify-center gap-2 rounded-xl bg-gradient-to-r from-blue-600 to-violet-600 py-3 font-semibold text-white transition hover:scale-[1.02]"
            >
              <Send size={18} />
              Send Reset Link
            </button>
          </form>
        ) : (
          <div className="py-4 text-center">
            <div className="mx-auto flex h-16 w-16 items-center justify-center rounded-full bg-emerald-500/20 text-emerald-400">
              <CheckCircle2 size={34} />
            </div>

            <h2 className="mt-5 text-2xl font-bold text-white">
              Email Sent
            </h2>

            <p className="mt-3 text-sm leading-relaxed text-slate-300">
              A password reset link has been sent to{" "}
              <span className="font-semibold text-white">
                {email}
              </span>
            </p>

            <div className="mt-6 flex items-center justify-center gap-2 text-sm text-blue-300">
              <Sparkles size={16} />
              Redirecting to Login...
            </div>
          </div>
        )}

        <div className="mt-8 border-t border-white/10 pt-6">
          <Link
            to="/login"
            className="flex items-center justify-center gap-2 text-sm font-semibold text-slate-300 transition hover:text-white"
          >
            <ArrowLeft size={16} />
            Back to Login
          </Link>
        </div>
      </div>
    </div>
  );
}

export default ForgotPassword;